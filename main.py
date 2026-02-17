"""Simple offline RL pipeline: download dataset, train model, evaluate."""

import minari
import numpy as np
from sklearn.neural_network import MLPClassifier

DATASET_ID = "D4RL/minigrid/fourrooms-v0"

# --- 1. Download & inspect dataset ---

dataset = minari.load_dataset(DATASET_ID, download=True)
print(f"Dataset: {dataset.id}")
print(f"Total episodes: {dataset.total_episodes}")
print(f"Total steps: {dataset.total_steps}")
print()

# --- 2. Train behavioral cloning model ---

observations = []
actions = []

for episode in dataset.iterate_episodes():
    obs_images = episode.observations["image"][:-1]  # T+1 obs for T actions
    obs_dirs = episode.observations["direction"][:-1]
    # Flatten 7x7x3 image + direction into feature vector
    flat_images = obs_images.reshape(len(obs_images), -1)
    flat_obs = np.column_stack([flat_images, obs_dirs])
    observations.append(flat_obs)
    actions.append(episode.actions)

X = np.concatenate(observations)
y = np.concatenate(actions)

print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.1%}")
print()

# --- 3. Evaluate trained policy ---

env = dataset.recover_environment()
num_eval_episodes = 5
rewards = []

for i in range(num_eval_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        flat = np.concatenate([obs["image"].flatten(), [obs["direction"]]])
        action = int(model.predict(flat.reshape(1, -1))[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    print(f"Episode {i + 1}: reward = {total_reward:.0f}")

print(f"\nMean reward: {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
env.close()
