import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from .utils import encode_state, learned_reward

from .mouse import (
    initialize_grid_with_cheese_types,
    move,
    get_reward,
    GRID_SIZE,
    EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC_CHEESE,
    ACTIONS, ACTION_TO_DELTA
)

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * GRID_SIZE * GRID_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS)),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

# Initialize new policy net for retraining
policy_net_retrain = PolicyNet()
optimizer = optim.Adam(policy_net_retrain.parameters(), lr=1e-3)

gamma = 0.99
step_penalty = -0.2
entropy_beta = 0.01

# --- Moving baseline ---
baseline = 0.0
baseline_alpha = 0.9  # Higher = more smoothing

# Save original policy net and freeze it for KL penalty baseline
original_policy_net_baseline = PolicyNet()
original_policy_net_baseline.load_state_dict(policy_net_retrain.state_dict())
original_policy_net_baseline.eval()  # Freeze baseline policy
for param in original_policy_net_baseline.parameters():
    param.requires_grad = False

penalty_strength = 0.1  # You can tune this

def run_episode_with_learned_reward(mean_sc_skill, mean_oc_skill):
    grid, _, _, _ = initialize_grid_with_cheese_types()
    log_probs, rewards, entropies, kl_penalties = [], [], [], []
    success = False
    steps = 0
    max_steps = 50

    while steps < max_steps:
        state_tensor = encode_state(grid).unsqueeze(0)

        # Current policy output
        current_probs = policy_net_retrain(state_tensor)
        dist = Categorical(current_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        # Original policy output for penalty
        with torch.no_grad():
            original_probs = original_policy_net_baseline(state_tensor)

        # Compute KL divergence penalty per state
        kl_div = F.kl_div(current_probs.log(), original_probs, reduction='batchmean')

        action = ACTIONS[action_idx.item()]
        old_mouse_pos = tuple(np.argwhere(grid == MOUSE)[0])
        dr, dc = ACTION_TO_DELTA[action]
        new_pos = (old_mouse_pos[0] + dr, old_mouse_pos[1] + dc)

        if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and grid[new_pos] != WALL:
            cell_content = grid[new_pos]
            grid = move(action, grid)
        else:
            cell_content = WALL

        if cell_content == WALL:
            reward = -0.2
        else:
            reward = learned_reward(cell_content, mean_sc_skill, mean_oc_skill)

        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        kl_penalties.append(kl_div)

        if cell_content in [CHEESE, ORGANIC_CHEESE]:
            success = True
            break

        steps += 1

    return log_probs, rewards, entropies, kl_penalties, success

def train_with_penalty(mean_sc_skill, mean_oc_skill, training_logs, num_batches=400, batch_size=10):
    global baseline
    all_losses, all_success_rates, all_kl_values, all_baselines = [], [], [], []

    for batch in range(1, num_batches + 1):
        batch_loss = 0.0
        batch_successes = []
        batch_kl_vals = []
        batch_returns = []
        batch_episodes = []

        for _ in range(batch_size):
            log_probs, rewards, entropies, kl_penalties, success = run_episode_with_learned_reward(mean_sc_skill, mean_oc_skill)
            batch_episodes.append((log_probs, rewards, entropies, kl_penalties))
            batch_successes.append(1 if success else 0)
            batch_kl_vals.append(torch.stack(kl_penalties).mean().item())

            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
            batch_returns.append(G)

        batch_mean_return = np.mean(batch_returns)
        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * batch_mean_return

        for log_probs, rewards, entropies, kl_penalties in batch_episodes:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            adjusted_returns = returns - baseline
            if len(adjusted_returns) > 1:
                adjusted_returns = (adjusted_returns - adjusted_returns.mean()) / (adjusted_returns.std() + 1e-9)

            pg_loss = sum(-log_prob * Gt for log_prob, Gt in zip(log_probs, adjusted_returns))
            entropy_loss = -entropy_beta * torch.stack(entropies).mean()
            kl_loss = penalty_strength * torch.stack(kl_penalties).mean()

            loss = pg_loss + entropy_loss + kl_loss
            batch_loss += loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        avg_loss = batch_loss.item() / batch_size
        avg_success = np.mean(batch_successes) * 100
        avg_kl = np.mean(batch_kl_vals)

        all_losses.append(avg_loss)
        all_success_rates.append(avg_success)
        all_kl_values.append(avg_kl)
        all_baselines.append(baseline)

        if batch % 10 == 0:
            training_logs.append(f"Batch {batch}/{num_batches} | Avg Loss: {avg_loss:.4f} | "
                  f"Success Rate: {avg_success:.1f}% | Baseline: {baseline:.2f} | KL: {avg_kl:.6f}")

    torch.save({
        'model_state_dict': policy_net_retrain.state_dict(),
        'mean_sc_skill': mean_sc_skill,
        'mean_oc_skill': mean_oc_skill
    }, "project5/mouse_policy_retrained_with_penalty.pth")

    print("Retrained model saved as mouse_policy_retrained_with_penalty.pth")

def run_episode_full_updated(max_steps=50, step_penalty=-0.2):
    model_path='project5/mouse_policy_retrained_with_penalty.pth'
    checkpoint = torch.load(model_path, weights_only=False)
    mean_sc_skill = checkpoint['mean_sc_skill']
    mean_oc_skill = checkpoint['mean_oc_skill']
    policy_net_retrain.load_state_dict(checkpoint['model_state_dict'])
    policy_net_retrain.eval()
    grid, _, _, _ = initialize_grid_with_cheese_types()
    log_probs, rewards, entropies = [], [], []
    states = []
    actions_taken = []
    organic_cheese_count = 0
    standard_cheese_count = 0
    success = False
    steps = 0

    while steps < max_steps:
        state_tensor = encode_state(grid).unsqueeze(0)  # Add batch dim
        with torch.no_grad():  # No grad needed for inference
            probs = policy_net_retrain(state_tensor)
        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

        action = ACTIONS[action_idx.item()]
        old_mouse_pos = tuple(np.argwhere(grid == MOUSE)[0])
        dr, dc = ACTION_TO_DELTA[action]
        new_pos = (old_mouse_pos[0] + dr, old_mouse_pos[1] + dc)

        if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and grid[new_pos] != WALL and grid[new_pos] != TRAP:
            cell_content = grid[new_pos]
            grid = move(action, grid)
        else:
            cell_content = WALL

        if cell_content == WALL:
            reward = step_penalty
        else:
            reward = learned_reward(cell_content, mean_sc_skill, mean_oc_skill)
            if reward == -0.2:  # step_penalty fallback
                reward = step_penalty

        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        states.append(state_tensor.squeeze(0))  # Remove batch dim
        actions_taken.append(action_idx.item())

        if cell_content == ORGANIC_CHEESE:
            organic_cheese_count += 1
        elif cell_content == CHEESE:
            standard_cheese_count += 1

        if cell_content in [CHEESE, ORGANIC_CHEESE]:
            success = True
            break

        steps += 1

    trajectory = {
        'states': states,
        'actions': actions_taken,
        'log_probs': log_probs,
        'rewards': rewards,
        'organic_cheese_count': organic_cheese_count,
        'standard_cheese_count': standard_cheese_count,
        'success': success,
    }
    return trajectory


def generate_trajectory():
    while 1:
        traj = run_episode_full_updated()
        if traj['standard_cheese_count']:
            break
    return traj