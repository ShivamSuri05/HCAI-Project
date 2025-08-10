import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

from mouse import (
    initialize_grid_with_cheese_types,
    move,
    get_reward,
    GRID_SIZE,
    EMPTY, MOUSE, CHEESE, TRAP, WALL, ORGANIC_CHEESE,
    ACTIONS, ACTION_TO_DELTA
)

def encode_state(grid):
    channels = 6
    encoded = np.zeros((channels, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            encoded[grid[r, c], r, c] = 1.0
    return torch.tensor(encoded, dtype=torch.float32)

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

policy_net = PolicyNet()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
gamma = 0.99
step_penalty = -0.2
entropy_beta = 0.01
baseline = 0.0
baseline_alpha = 0.9

# Run one episode
def run_episode():
    grid, _, _, _ = initialize_grid_with_cheese_types()
    log_probs, rewards, entropies = [], [], []
    success = False
    steps = 0
    max_steps = 50

    while steps < max_steps:
        state_tensor = encode_state(grid).unsqueeze(0)
        probs = policy_net(state_tensor)
        dist = Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()

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
            reward = step_penalty
        else:
            reward = get_reward(new_pos, grid)
            if reward == -0.2:
                reward = step_penalty

        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)

        if cell_content in [CHEESE, ORGANIC_CHEESE]:
            success = True
            break

        steps += 1

    return log_probs, rewards, entropies, success

# N-batch REINFORCE training
def train(num_batches=400, batch_size=10):
    global baseline
    all_losses, all_success_rates = [], []

    for batch in range(1, num_batches + 1):
        batch_loss = 0.0
        batch_returns = []
        batch_successes = []

        # Collect N episodes
        batch_episodes = []
        for _ in range(batch_size):
            log_probs, rewards, entropies, success = run_episode()
            batch_episodes.append((log_probs, rewards, entropies))
            batch_successes.append(1 if success else 0)

            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
            batch_returns.append(G)

        batch_mean_return = np.mean(batch_returns)
        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * batch_mean_return

        # Compute loss for the batch
        for log_probs, rewards, entropies in batch_episodes:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            adjusted_returns = returns - baseline

            # Normalize
            if len(adjusted_returns) > 1:
                adjusted_returns = (adjusted_returns - adjusted_returns.mean()) / (adjusted_returns.std() + 1e-9)

            pg_loss = sum(-log_prob * Gt for log_prob, Gt in zip(log_probs, adjusted_returns))
            entropy_loss = -entropy_beta * torch.stack(entropies).mean()
            loss = pg_loss + entropy_loss
            batch_loss += loss

        # Gradient step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        avg_loss = batch_loss.item() / batch_size
        avg_success = np.mean(batch_successes) * 100
        all_losses.append(avg_loss)
        all_success_rates.append(avg_success)

        # Logging
        if batch % 10 == 0:
            print(f"Batch {batch}/{num_batches} | Avg Loss (last batch): {avg_loss:.4f} | "
                  f"Success Rate: {avg_success:.1f}% | Baseline: {baseline:.2f}")


    torch.save(policy_net.state_dict(), "project5/mouse_policy_task1.pth")
    print("Model saved as mouse_policy_task1.pth")

    # Plot Loss & Success Rate
    episodes_x = np.arange(len(all_losses))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(all_losses)
    plt.title("Batch Loss per update")
    plt.xlabel("Batch update #")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    win = 10
    success_pct = [np.mean(all_success_rates[max(0,i-win+1):i+1]) for i in range(len(all_success_rates))]
    plt.plot(success_pct)
    plt.title(f"Success rate (rolling window={win} batches)")
    plt.xlabel("Batch update #")
    plt.ylabel("Success rate %")

    plt.tight_layout()
    plt.show()

# Run training
train(num_batches=10, batch_size=2)