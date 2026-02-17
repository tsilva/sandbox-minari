"""Offline RL with DiscreteCQL: train on Minari dataset, evaluate in environment."""

import d3rlpy
import minari
import numpy as np
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm

DATASET_ID = "D4RL/minigrid/fourrooms-v0"
NUM_EVAL_EPISODES = 20
N_STEPS = 50000
N_STEPS_PER_EPOCH = 5000


def flatten_obs(image, direction):
    """Flatten 7x7x3 image + direction into a single vector."""
    return np.concatenate([image.reshape(-1), np.atleast_1d(direction)])


# --- 1. Download & inspect dataset ---

print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

dataset = minari.load_dataset(DATASET_ID, download=True)

print(f"  Dataset:      {dataset.id}")
print(f"  Environment:  {dataset.env_spec.id}")
print(f"  Episodes:     {dataset.total_episodes}")
print(f"  Total steps:  {dataset.total_steps}")
print()

# --- 2. Prepare data for d3rlpy ---

print("=" * 60)
print("STEP 2: Preparing data for d3rlpy")
print("=" * 60)

observations = []
actions = []
rewards = []
next_observations = []
terminals = []

for episode in tqdm(dataset.iterate_episodes(), total=dataset.total_episodes, desc="  Episodes"):
    images = episode.observations["image"]
    directions = episode.observations["direction"]
    n_steps = len(episode.actions)

    for t in range(n_steps):
        observations.append(flatten_obs(images[t], directions[t]))
        actions.append(episode.actions[t])
        rewards.append(episode.rewards[t])
        next_observations.append(flatten_obs(images[t + 1], directions[t + 1]))
        is_terminal = (t == n_steps - 1) and episode.terminations[-1]
        terminals.append(is_terminal)

observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.int64)
rewards = np.array(rewards, dtype=np.float32)
next_observations = np.array(next_observations, dtype=np.float32)
terminals = np.array(terminals, dtype=np.float32)

mdp_dataset = MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
)

print(f"\n  Observations: {observations.shape} (7x7x3 image flattened + direction)")
print(f"  Actions:      {actions.shape} (discrete: {len(np.unique(actions))} unique)")
print(f"  Terminals:    {int(terminals.sum())} terminal states")
print()

# --- 3. Train DiscreteCQL ---

print("=" * 60)
print("STEP 3: Training DiscreteCQL")
print("=" * 60)

cql = d3rlpy.algos.DiscreteCQLConfig(
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
).create(device="cpu")

cql.fit(
    mdp_dataset,
    n_steps=N_STEPS,
    n_steps_per_epoch=N_STEPS_PER_EPOCH,
    show_progress=True,
)

print()

# --- 4. Evaluate trained policy ---

print("=" * 60)
print(f"STEP 4: Evaluating for {NUM_EVAL_EPISODES} episodes in environment")
print("=" * 60)

env = dataset.recover_environment(render_mode="human")
eval_rewards = []
episode_steps = []

for i in tqdm(range(NUM_EVAL_EPISODES), desc="  Evaluating", unit="ep"):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        flat = flatten_obs(obs["image"], obs["direction"]).reshape(1, -1)
        action = int(cql.predict(flat)[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    eval_rewards.append(total_reward)
    episode_steps.append(steps)
    result = "reached goal" if total_reward > 0 else "timed out"
    tqdm.write(f"  Episode {i + 1:2d}: {result:12s} | {steps:4d} steps | reward={total_reward:.3f}")

env.close()

# --- Summary ---

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
n_success = sum(1 for r in eval_rewards if r > 0)
print(f"  Algorithm:    DiscreteCQL (d3rlpy)")
print(f"  Training:     {N_STEPS} steps ({N_STEPS // N_STEPS_PER_EPOCH} epochs)")
print(f"  Success rate: {n_success}/{len(eval_rewards)} episodes ({n_success / len(eval_rewards):.0%})")
print(f"  Mean reward:  {np.mean(eval_rewards):.3f} (+/- {np.std(eval_rewards):.3f})")
mean_success_steps = np.mean([s for s, r in zip(episode_steps, eval_rewards) if r > 0]) if n_success > 0 else 0
print(f"  Mean steps:   {np.mean(episode_steps):.0f} (successes: {mean_success_steps:.0f})")
print()
print("  Compare with main.py (behavioral cloning) to see the difference")
print("  between imitation learning and conservative offline RL.")
