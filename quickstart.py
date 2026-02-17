"""Quickstart: explore offline RL datasets with Minari."""

import minari

# 1. List available remote datasets
remote_datasets = minari.list_remote_datasets()
dataset_names = sorted(remote_datasets.keys())
print(f"Available remote datasets: {len(dataset_names)}")
for name in dataset_names[:10]:
    print(f"  - {name}")
print()

# 2. Download and load a dataset
DATASET_ID = "D4RL/door/human-v2"
print(f"Loading dataset: {DATASET_ID}")
dataset = minari.load_dataset(DATASET_ID, download=True)

# 3. Print dataset metadata
print(f"  Observation space: {dataset.observation_space}")
print(f"  Action space:      {dataset.action_space}")
print(f"  Total episodes:    {dataset.total_episodes}")
print(f"  Total steps:       {dataset.total_steps}")
print()

# 4. Iterate first 3 episodes
print("First 3 episodes:")
for episode in dataset.iterate_episodes(episode_indices=range(3)):
    total_reward = episode.rewards.sum()
    n_steps = len(episode.rewards)
    print(f"  Episode {episode.id}: {n_steps} steps, total reward: {total_reward:.2f}")
