"""Simple offline RL pipeline: download dataset, train model, evaluate."""

import minari
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

DATASET_ID = "D4RL/minigrid/fourrooms-v0"
TRAIN_SPLIT = 0.8
VAL_FRACTION = 0.15  # fraction of train used for validation
MAX_EPOCHS = 500
PATIENCE = 20
NUM_EVAL_EPISODES = 20


def extract_features(episode):
    """Flatten 7x7x3 image + direction into a feature vector per step."""
    obs_images = episode.observations["image"][:-1]  # T+1 obs for T actions
    obs_dirs = episode.observations["direction"][:-1]
    flat_images = obs_images.reshape(len(obs_images), -1)
    return np.column_stack([flat_images, obs_dirs])


def obs_to_features(obs):
    """Convert a single env observation to a feature vector."""
    return np.concatenate([obs["image"].flatten(), [obs["direction"]]])


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

# --- 2. Extract and split data ---

print("=" * 60)
print("STEP 2: Extracting data & splitting train/test by episode")
print("=" * 60)

all_features = []
all_actions = []

for episode in tqdm(dataset.iterate_episodes(), total=dataset.total_episodes, desc="  Episodes"):
    all_features.append(extract_features(episode))
    all_actions.append(episode.actions)

# Split at episode level to avoid leaking within-episode correlations
rng = np.random.RandomState(42)
n_episodes = len(all_features)
indices = rng.permutation(n_episodes)
split = int(TRAIN_SPLIT * n_episodes)
train_idx, test_idx = indices[:split], indices[split:]

X_train = np.concatenate([all_features[i] for i in train_idx])
y_train = np.concatenate([all_actions[i] for i in train_idx])
X_test = np.concatenate([all_features[i] for i in test_idx])
y_test = np.concatenate([all_actions[i] for i in test_idx])

print(f"\n  Features:       {X_train.shape[1]} (7x7x3 image flattened + direction)")
print(f"  Train episodes: {len(train_idx)} ({X_train.shape[0]} steps)")
print(f"  Test episodes:  {len(test_idx)} ({X_test.shape[0]} steps)")

unique, counts = np.unique(y_train, return_counts=True)
print(f"  Action dist:    {dict(zip(unique, counts))} (0=left, 1=right, 2=forward)")
print()

# Further split train into train/val for early stopping
val_size = int(VAL_FRACTION * len(X_train))
val_idx = rng.permutation(len(X_train))
X_val, y_val = X_train[val_idx[:val_size]], y_train[val_idx[:val_size]]
X_train_inner, y_train_inner = X_train[val_idx[val_size:]], y_train[val_idx[val_size:]]

print(f"  Train (inner):  {X_train_inner.shape[0]} steps")
print(f"  Validation:     {X_val.shape[0]} steps")
print()

# --- 3. Train with early stopping ---

print("=" * 60)
print("STEP 3: Training MLPClassifier(64, 32) with early stopping")
print("=" * 60)

model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
classes = np.unique(y_train)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

pbar = tqdm(range(1, MAX_EPOCHS + 1), desc="  Training", unit="epoch")
for epoch in pbar:
    model.partial_fit(X_train_inner, y_train_inner, classes=classes)

    val_proba = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_proba, labels=classes)
    val_acc = np.mean(model.predict(X_val) == y_val)

    pbar.set_postfix(
        train_loss=f"{model.loss_:.4f}",
        val_loss=f"{val_loss:.4f}",
        val_acc=f"{val_acc:.1%}",
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            tqdm.write(f"\n  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

train_acc = model.score(X_train_inner, y_train_inner)
val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)

print(f"\n  Train accuracy: {train_acc:.1%}")
print(f"  Val accuracy:   {val_acc:.1%}")
print(f"  Test accuracy:  {test_acc:.1%}")

gap = train_acc - test_acc
if gap > 0.05:
    print(f"  Overfit gap:    {gap:.1%} (train - test)")
print()

# --- 4. Evaluate trained policy ---

print("=" * 60)
print(f"STEP 4: Evaluating for {NUM_EVAL_EPISODES} episodes in environment")
print("=" * 60)

env = dataset.recover_environment(render_mode="human")
rewards = []
episode_steps = []

for i in tqdm(range(NUM_EVAL_EPISODES), desc="  Evaluating", unit="ep"):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        flat = obs_to_features(obs)
        action = int(model.predict(flat.reshape(1, -1))[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    rewards.append(total_reward)
    episode_steps.append(steps)
    result = "reached goal" if total_reward > 0 else "timed out"
    tqdm.write(f"  Episode {i + 1:2d}: {result:12s} | {steps:4d} steps | reward={total_reward:.3f}")

env.close()

# --- Summary ---

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
n_success = sum(1 for r in rewards if r > 0)
print(f"  Test accuracy:  {test_acc:.1%} (held-out episodes)")
print(f"  Success rate:   {n_success}/{len(rewards)} episodes ({n_success/len(rewards):.0%})")
print(f"  Mean reward:    {np.mean(rewards):.3f} (+/- {np.std(rewards):.3f})")
print(f"  Mean steps:     {np.mean(episode_steps):.0f} (successes: {np.mean([s for s, r in zip(episode_steps, rewards) if r > 0]) if n_success > 0 else 0:.0f})")
