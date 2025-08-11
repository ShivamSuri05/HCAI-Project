import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .mouse import (
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
policy_net.load_state_dict(torch.load("project5/mouse_policy_task1.pth", map_location=torch.device('cpu')))
policy_net.eval()
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


def generate_filtered_trajectory_pairs(num_pairs):
    pairs = []
    while len(pairs) < num_pairs:
        traj1 = run_episode_full()
        traj2 = run_episode_full()

        # Check if one has organic cheese and the other doesn't
        if traj1['organic_cheese_count'] != traj2['organic_cheese_count']:
            pairs.append((traj1, traj2))
    return pairs

def sort_pairs_preserving_organic_diff(pairs):
    # Separate traj1 and traj2 by organic_cheese_count
    traj_with_0 = []
    traj_with_1 = []

    # Flatten pairs into two lists, track which traj belongs where
    for traj1, traj2 in pairs:
        # traj1 and traj2 have different organic_cheese_count by design
        # So assign traj1 and traj2 accordingly
        if traj1['organic_cheese_count'] == 0:
            traj_with_0.append(traj1)
            traj_with_1.append(traj2)
        else:
            traj_with_0.append(traj2)
            traj_with_1.append(traj1)

    # Sort both lists descending by total rewards
    traj_with_0_sorted = sorted(traj_with_0, key=lambda t: sum(t['rewards']), reverse=True)
    traj_with_1_sorted = sorted(traj_with_1, key=lambda t: sum(t['rewards']), reverse=True)

    # Zip back to pairs ensuring one from each list
    sorted_pairs = list(zip(traj_with_0_sorted, traj_with_1_sorted))
    return sorted_pairs

def convert_to_pairs(all_feedback):
    pairs = []
    for feedback in all_feedback:
        pairs.append((feedback['trajectory1']['frames'], feedback['trajectory2']['frames'], feedback['choice']))
    return pairs

def bradley_terry(preferences, n):
    """
    preferences: list of (winner_id, loser_id)
    n: number of trajectories
    Returns: numpy array theta of estimated skills for each trajectory.
    """
    def neg_log_likelihood(theta):
        total = 0
        for winner, loser in preferences:
            diff = theta[winner] - theta[loser]
            total += np.log(1 + np.exp(-diff))
        return total

    def grad(theta):
        gradient = np.zeros_like(theta)
        for winner, loser in preferences:
            diff = theta[winner] - theta[loser]
            exp_neg = np.exp(-diff)
            gradient[winner] -= exp_neg / (1 + exp_neg)
            gradient[loser] += exp_neg / (1 + exp_neg)
        return gradient

    # Fix first skill to 0 (to remove scale ambiguity)
    def constraint(theta):
        return theta[0]

    cons = {'type': 'eq', 'fun': constraint}
    theta0 = np.zeros(n)
    result = minimize(neg_log_likelihood, theta0, jac=grad, constraints=cons, method='SLSQP')
    return result.x

def learned_reward(cell_content, mean_sc_skill, mean_oc_skill):
    if cell_content == CHEESE:  # replace with your constant or enum
        return mean_sc_skill
    elif cell_content == ORGANIC_CHEESE:
        return mean_oc_skill
    elif cell_content == TRAP:
        return -50
    else:
        return -0.2

def compute_mean_skills(theta, sc_list, oc_list):
    start = sc_list[0]
    if len(sc_list)>2:
        start = sc_list[2]
    sc_list[0] = start
    sc_skills = [theta[idx] for idx in sc_list]
    oc_skills = [theta[idx] for idx in oc_list]

    mean_sc_skill = np.mean(sc_skills) if sc_skills else 0.0
    mean_oc_skill = np.mean(oc_skills) if oc_skills else 0.0

    return mean_sc_skill, mean_oc_skill

def collect_preferences(trajectory_pairs):
    preferences = []  # List of (winner_id, loser_id)
    traj_id_map = {}  # Map trajectory object id to unique int id
    sc_list, oc_list = [] , []
    id_counter = 0

    def get_id(traj):
        nonlocal id_counter
        traj_key = id(traj)
        if traj_key not in traj_id_map:
            traj_id_map[traj_key] = id_counter
            id_counter += 1
        return traj_id_map[traj_key]

    for i, (traj1, traj2, choice) in enumerate(trajectory_pairs, 1):

        id1 = get_id(traj1)
        id2 = get_id(traj2)

        if traj2['organic_cheese_count']:
            sc_list.append(id1)
            oc_list.append(id2)
        else:
            sc_list.append(id2)
            oc_list.append(id1)

        if choice == 1:
            preferences.append((id1, id2))
        else:
            preferences.append((id2, id1))


    return preferences, traj_id_map, sc_list, oc_list


def apply_last_action_and_get_final_grid(traj):
    last_grid = decode_state_tensor(traj['states'][-1])
    last_action = traj['actions'][-1]

    # Map action index to direction delta, adjust as per your ACTIONS/ACTION_TO_DELTA
    # Assuming ACTIONS = ['up', 'down', 'left', 'right']
    ACTION_TO_DELTA = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    dr, dc = ACTION_TO_DELTA[last_action]

    mouse_pos = np.argwhere(last_grid == MOUSE)
    if len(mouse_pos) == 0:
        # Mouse not found? return last grid as fallback
        return last_grid
    r, c = mouse_pos[0]

    new_r, new_c = r + dr, c + dc
    if 0 <= new_r < last_grid.shape[0] and 0 <= new_c < last_grid.shape[1]:
        # Remove mouse from old position
        last_grid[r, c] = EMPTY
        # Put mouse on new position (overwriting cheese if any)
        last_grid[new_r, new_c] = MOUSE

    return last_grid

def run_episode_full():
    grid, _, _, _ = initialize_grid_with_cheese_types()
    log_probs, rewards, entropies = [], [], []
    states = []
    actions_taken = []
    organic_cheese_count = 0
    standard_cheese_count = 0
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

        if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and grid[new_pos] != WALL and grid[new_pos] != TRAP:
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
        states.append(state_tensor.squeeze(0))  # Store state tensor without batch dim
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


def decode_state_tensor(state_tensor):
    """
    Converts one-hot encoded tensor state (6 x 5 x 5) to a 2D grid of ints (5 x 5).
    """
    # state_tensor shape: [channels=6, height=5, width=5]
    # Get the index of the channel with max value at each cell
    grid = torch.argmax(state_tensor, dim=0)
    # Convert to numpy array of ints
    return grid.numpy().astype(int)