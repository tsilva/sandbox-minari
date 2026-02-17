"""Simple offline RL pipeline: download dataset, train model, evaluate."""

import minari
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

DATASET_ID = "D4RL/minigrid/fourrooms-v0"
NUM_EPOCHS = 200
NUM_EVAL_EPISODES = 5

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

# --- 2. Extract training data ---

print("=" * 60)
print("STEP 2: Extracting (observation, action) pairs")
print("=" * 60)

observations = []
actions = []

for episode in tqdm(dataset.iterate_episodes(), total=dataset.total_episodes, desc="  Episodes"):
    obs_images = episode.observations["image"][:-1]  # T+1 obs for T actions
    obs_dirs = episode.observations["direction"][:-1]
    # Flatten 7x7x3 image + direction into feature vector
    flat_images = obs_images.reshape(len(obs_images), -1)
    flat_obs = np.column_stack([flat_images, obs_dirs])
    observations.append(flat_obs)
    actions.append(episode.actions)

X = np.concatenate(observations)
y = np.concatenate(actions)

unique_actions, action_counts = np.unique(y, return_counts=True)
print(f"\n  Samples:  {X.shape[0]}")
print(f"  Features: {X.shape[1]} (7x7x3 image flattened + direction)")
print(f"  Actions:  {dict(zip(unique_actions, action_counts))}")
print(f"            (0=left, 1=right, 2=forward)")
print()

# --- 3. Train behavioral cloning model ---

print("=" * 60)
print("STEP 3: Training MLPClassifier(64, 32)")
print("=" * 60)

model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
classes = np.unique(y)

pbar = tqdm(range(1, NUM_EPOCHS + 1), desc="  Training", unit="epoch")
for epoch in pbar:
    model.partial_fit(X, y, classes=classes)
    pbar.set_postfix(loss=f"{model.loss_:.4f}")

train_acc = model.score(X, y)
print(f"\n  Final loss:     {model.loss_:.4f}")
print(f"  Train accuracy: {train_acc:.1%}")
print()

# --- 4. Evaluate trained policy ---

print("=" * 60)
print(f"STEP 4: Evaluating for {NUM_EVAL_EPISODES} episodes")
print("=" * 60)

env = dataset.recover_environment(render_mode="human")
rewards = []

for i in tqdm(range(NUM_EVAL_EPISODES), desc="  Evaluating", unit="ep"):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        flat = np.concatenate([obs["image"].flatten(), [obs["direction"]]])
        action = int(model.predict(flat.reshape(1, -1))[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    rewards.append(total_reward)
    result = "reached goal" if total_reward > 0 else "timed out"
    tqdm.write(f"  Episode {i + 1}: {result} in {steps} steps (reward={total_reward:.3f})")

env.close()

# --- Summary ---

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
print(f"  Success rate: {success_rate:.0%} ({sum(1 for r in rewards if r > 0)}/{len(rewards)} episodes)")
print(f"  Mean reward:  {np.mean(rewards):.3f} (+/- {np.std(rewards):.3f})")
