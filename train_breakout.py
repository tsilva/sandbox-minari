"""Train a behavioral-cloning policy on Minari's Breakout expert dataset."""

import argparse
from pathlib import Path

import d3rlpy
import minari
import numpy as np
from d3rlpy.algos import DiscreteBCConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import PixelEncoderFactory
from PIL import Image
from tqdm import tqdm

DATASET_ID = "atari/breakout/expert-v0"
TRAIN_SPLIT = 0.8
VAL_FRACTION = 0.25
SCREEN_SIZE = 84
STACK_SIZE = 4
N_STEPS = 3000
N_STEPS_PER_EPOCH = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
FEATURE_SIZE = 256
NUM_EVAL_EPISODES = 5
MAX_EVAL_STEPS = 5000
DEFAULT_MODEL_PATH = "models/breakout_bc.d3"
PATIENCE = 5
MIN_DELTA = 0.001
SEED = 42


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--screen-size", type=int, default=SCREEN_SIZE)
    parser.add_argument("--stack-size", type=int, default=STACK_SIZE)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-steps-per-epoch", type=int, default=N_STEPS_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--feature-size", type=int, default=FEATURE_SIZE)
    parser.add_argument("--num-eval-episodes", type=int, default=NUM_EVAL_EPISODES)
    parser.add_argument("--max-eval-steps", type=int, default=MAX_EVAL_STEPS)
    parser.add_argument("--save-model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--load-model-path")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", "none"],
        default="none",
        help="Use 'human' to watch evaluation locally.",
    )
    return parser.parse_args()


def preprocess_frame(frame, screen_size):
    """Crop, de-flicker, grayscale, and resize a raw Atari frame."""
    image = Image.fromarray(frame)
    image = image.crop((0, 34, 160, 194)).convert("L")
    image = image.resize((screen_size, screen_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def preprocess_episode_frames(observations, screen_size):
    """Match common Atari preprocessing using max-pooling over adjacent frames."""
    processed = np.empty((len(observations), screen_size, screen_size), dtype=np.uint8)
    previous_frame = observations[0]

    for index, frame in enumerate(observations):
        pooled = np.maximum(previous_frame, frame) if index else frame
        processed[index] = preprocess_frame(pooled, screen_size)
        previous_frame = frame

    return processed


def build_episode_buffer(episode, screen_size, stack_size):
    """Convert one Minari episode into stacked image observations."""
    frames = preprocess_episode_frames(episode.observations, screen_size)
    num_steps = len(episode.actions)
    stacked_obs = np.empty((num_steps, stack_size, screen_size, screen_size), dtype=np.uint8)
    history = np.repeat(frames[0:1], stack_size, axis=0)

    for step in range(num_steps):
        stacked_obs[step] = history
        history = np.concatenate([history[1:], frames[step + 1 : step + 2]], axis=0)

    terminals = np.zeros(num_steps, dtype=np.float32)
    timeouts = np.zeros(num_steps, dtype=np.float32)
    terminals[-1] = float(episode.terminations[-1])
    timeouts[-1] = float(episode.truncations[-1])

    return {
        "observations": stacked_obs,
        "actions": episode.actions.astype(np.int64),
        "rewards": episode.rewards.astype(np.float32),
        "terminals": terminals,
        "timeouts": timeouts,
        "return": float(episode.rewards.sum()),
    }


def concatenate_episode_buffers(episode_buffers, indices):
    """Concatenate per-episode arrays into one training or test split."""
    selected = [episode_buffers[index] for index in indices]
    return {
        "observations": np.concatenate([item["observations"] for item in selected], axis=0),
        "actions": np.concatenate([item["actions"] for item in selected], axis=0),
        "rewards": np.concatenate([item["rewards"] for item in selected], axis=0),
        "terminals": np.concatenate([item["terminals"] for item in selected], axis=0),
        "timeouts": np.concatenate([item["timeouts"] for item in selected], axis=0),
    }


def predict_in_batches(policy, observations, batch_size=128):
    """Run policy predictions without materializing one giant tensor."""
    predictions = []
    for start in range(0, len(observations), batch_size):
        stop = start + batch_size
        predictions.append(policy.predict(observations[start:stop]))
    return np.concatenate(predictions, axis=0)


def accuracy(policy, data):
    """Compute action prediction accuracy for one split."""
    predictions = predict_in_batches(policy, data["observations"])
    return float(np.mean(predictions == data["actions"]))


def make_eval_env(dataset, render_mode):
    """Recover the Atari environment, optionally with rendering enabled."""
    kwargs = {} if render_mode == "none" else {"render_mode": render_mode}
    try:
        return dataset.recover_environment(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Could not recover ALE/Breakout-v5. Install project deps with `uv sync` "
            "and make Atari ROMs available if Gymnasium prompts for them."
        ) from exc


def reset_history(observation, screen_size, stack_size):
    """Initialize the frame stack from the first observation."""
    frame = preprocess_frame(observation, screen_size)
    return np.repeat(frame[None, ...], stack_size, axis=0)


def update_history(history, previous_observation, observation, screen_size):
    """Append the next preprocessed frame using the same de-flickering as training."""
    pooled = np.maximum(previous_observation, observation)
    next_frame = preprocess_frame(pooled, screen_size)
    return np.concatenate([history[1:], next_frame[None, ...]], axis=0)


def step_env(env, action, history, previous_observation, screen_size):
    """Apply an environment step and keep frame history aligned with training."""
    observation, reward, terminated, truncated, info = env.step(action)
    history = update_history(
        history,
        previous_observation=previous_observation,
        observation=observation,
        screen_size=screen_size,
    )
    return observation, reward, terminated, truncated, info, history


def find_fire_action(env):
    """Return the FIRE action index if the environment exposes one."""
    action_meanings = getattr(env.unwrapped, "get_action_meanings", lambda: [])()
    for index, meaning in enumerate(action_meanings):
        if meaning == "FIRE":
            return index
    return None


def create_policy(args):
    """Create the policy with the configured pixel encoder."""
    return DiscreteBCConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_factory=PixelEncoderFactory(feature_size=args.feature_size),
    ).create(device=args.device)


def load_saved_policy(args, model_path):
    """Instantiate the policy and load weights from disk."""
    try:
        policy = d3rlpy.load_learnable(str(model_path), device=args.device)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the saved model. Check that the checkpoint exists and that "
            "it was created by a compatible d3rlpy version."
        ) from exc
    return policy


def split_episode_indices(num_episodes, train_split, val_fraction, seed):
    """Split episodes into train, validation, and test sets."""
    if num_episodes < 3:
        raise ValueError("Need at least 3 episodes to create train/val/test splits.")

    rng = np.random.default_rng(seed)
    episode_indices = rng.permutation(num_episodes)

    train_val_count = int(train_split * num_episodes)
    train_val_count = min(max(train_val_count, 2), num_episodes - 1)
    val_count = int(round(val_fraction * train_val_count))
    val_count = min(max(val_count, 1), train_val_count - 1)
    train_count = train_val_count - val_count

    train_indices = episode_indices[:train_count]
    val_indices = episode_indices[train_count:train_val_count]
    test_indices = episode_indices[train_val_count:]
    return train_indices, val_indices, test_indices


def main():
    """Load dataset, train policy, and evaluate in the live environment."""
    args = parse_args()
    model_path = Path(args.load_model_path or args.save_model_path)

    if args.eval_only and not args.load_model_path:
        raise ValueError("--eval-only requires --load-model-path.")

    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)

    dataset = minari.load_dataset(args.dataset_id, download=True)

    print(f"  Dataset:      {dataset.id}")
    print(f"  Environment:  {dataset.env_spec.id}")
    print(f"  Episodes:     {dataset.total_episodes}")
    print(f"  Total steps:  {dataset.total_steps}")
    print(f"  Observation:  {dataset.observation_space}")
    print(f"  Action space: {dataset.action_space}")
    print()

    if args.eval_only:
        print("=" * 60)
        print("STEP 2: Loading saved policy")
        print("=" * 60)
        print(f"  Model path:    {model_path}")
        print(f"  Screen size:   {args.screen_size}")
        print(f"  Frame stack:   {args.stack_size}")
        print(f"  Feature size:  {args.feature_size}")
        print()
        policy = load_saved_policy(args, model_path)
        test_accuracy = None
    else:
        print("=" * 60)
        print("STEP 2: Preparing stacked Atari observations")
        print("=" * 60)

        episode_buffers = []
        for episode in tqdm(
            dataset.iterate_episodes(),
            total=dataset.total_episodes,
            desc="  Episodes",
        ):
            episode_buffers.append(build_episode_buffer(episode, args.screen_size, args.stack_size))

        train_indices, val_indices, test_indices = split_episode_indices(
            num_episodes=len(episode_buffers),
            train_split=args.train_split,
            val_fraction=args.val_fraction,
            seed=SEED,
        )

        train_data = concatenate_episode_buffers(episode_buffers, train_indices)
        val_data = concatenate_episode_buffers(episode_buffers, val_indices)
        test_data = concatenate_episode_buffers(episode_buffers, test_indices)

        unique_actions, counts = np.unique(train_data["actions"], return_counts=True)
        print(f"\n  Screen size:    {args.screen_size}x{args.screen_size} grayscale")
        print(f"  Frame stack:    {args.stack_size}")
        print(f"  Train episodes: {len(train_indices)} ({len(train_data['actions'])} steps)")
        print(f"  Val episodes:   {len(val_indices)} ({len(val_data['actions'])} steps)")
        print(f"  Test episodes:  {len(test_indices)} ({len(test_data['actions'])} steps)")
        mean_dataset_return = np.mean([item["return"] for item in episode_buffers])
        print(f"  Dataset return: {mean_dataset_return:.1f} avg reward/episode")
        print(f"  Action dist:    {dict(zip(unique_actions.tolist(), counts.tolist()))}")
        print()

        train_dataset = MDPDataset(
            observations=train_data["observations"],
            actions=train_data["actions"],
            rewards=train_data["rewards"],
            terminals=train_data["terminals"],
            timeouts=train_data["timeouts"],
        )

        print("=" * 60)
        print("STEP 3: Training DiscreteBC with early stopping on val accuracy")
        print("=" * 60)

        policy = create_policy(args)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        best_val_accuracy = float("-inf")
        best_epoch = 0
        best_step = 0
        patience_counter = 0

        for epoch, metrics in policy.fitter(
            train_dataset,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            experiment_name="breakout_bc",
            show_progress=True,
        ):
            train_accuracy = accuracy(policy, train_data)
            val_accuracy = accuracy(policy, val_data)
            total_step = epoch * args.n_steps_per_epoch
            improved = val_accuracy > (best_val_accuracy + args.min_delta)

            tqdm.write(
                "  Epoch "
                f"{epoch:2d}: loss={metrics['loss']:.4f} | "
                f"train_acc={train_accuracy:.1%} | val_acc={val_accuracy:.1%}"
            )

            if improved:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_step = total_step
                patience_counter = 0
                policy.save(str(model_path))
                tqdm.write(
                    f"    New best val accuracy; saved checkpoint to {model_path}"
                )
            else:
                patience_counter += 1
                tqdm.write(
                    f"    No improvement for {patience_counter}/{args.patience} epoch(s)"
                )
                if patience_counter >= args.patience:
                    tqdm.write(
                        f"    Early stopping at epoch {epoch}; best epoch was {best_epoch}"
                    )
                    break

        print(f"\n  Saved model:    {model_path}")
        print(f"  Best epoch:     {best_epoch}")
        print(f"  Best val acc:   {best_val_accuracy:.1%}")

        policy = load_saved_policy(args, model_path)
        train_accuracy = accuracy(policy, train_data)
        test_accuracy = accuracy(policy, test_data)
        print(f"  Train accuracy: {train_accuracy:.1%}")
        print(f"  Test accuracy:  {test_accuracy:.1%}")
        print()

    print("=" * 60)
    print(f"STEP 4: Evaluating for {args.num_eval_episodes} episodes in environment")
    print("=" * 60)

    env = make_eval_env(dataset, args.render_mode)
    fire_action = find_fire_action(env)
    rewards = []
    episode_steps = []

    for episode_number in tqdm(range(args.num_eval_episodes), desc="  Evaluating", unit="ep"):
        observation, info = env.reset()
        previous_observation = observation
        history = reset_history(observation, args.screen_size, args.stack_size)
        lives = info.get("lives")
        total_reward = 0.0
        steps = 0
        done = False

        if fire_action is not None:
            observation, reward, terminated, truncated, info, history = step_env(
                env=env,
                action=fire_action,
                history=history,
                previous_observation=previous_observation,
                screen_size=args.screen_size,
            )
            previous_observation = observation
            total_reward += reward
            steps += 1
            done = terminated or truncated
            lives = info.get("lives", lives)

        while not done:
            action = int(policy.predict(history[None, ...])[0])
            observation, reward, terminated, truncated, info, history = step_env(
                env=env,
                action=action,
                history=history,
                previous_observation=previous_observation,
                screen_size=args.screen_size,
            )
            total_reward += reward
            steps += 1
            done = terminated or truncated
            previous_observation = observation
            current_lives = info.get("lives", lives)

            if (
                not done
                and fire_action is not None
                and lives is not None
                and current_lives < lives
            ):
                observation, reward, terminated, truncated, info, history = step_env(
                    env=env,
                    action=fire_action,
                    history=history,
                    previous_observation=previous_observation,
                    screen_size=args.screen_size,
                )
                previous_observation = observation
                total_reward += reward
                steps += 1
                done = terminated or truncated
                current_lives = info.get("lives", current_lives)

            lives = current_lives

            if steps >= args.max_eval_steps:
                tqdm.write(
                    "  Episode "
                    f"{episode_number + 1:2d}: reached max_eval_steps={args.max_eval_steps}"
                )
                break

        rewards.append(total_reward)
        episode_steps.append(steps)
        tqdm.write(
            f"  Episode {episode_number + 1:2d}: reward={total_reward:6.1f} | steps={steps:4d}"
        )

    env.close()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Algorithm:      DiscreteBC (d3rlpy)")
    print(f"  Model path:     {model_path}")
    if not args.eval_only:
        print(f"  Max steps:      {args.n_steps}")
        print(f"  Best step:      {best_step}")
        print(f"  Best epoch:     {best_epoch}")
        print(f"  Best val acc:   {best_val_accuracy:.1%}")
    if test_accuracy is not None:
        print(f"  Test accuracy:  {test_accuracy:.1%} (held-out episodes)")
    print(f"  Mean reward:    {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
    print(f"  Mean steps:     {np.mean(episode_steps):.0f}")


if __name__ == "__main__":
    main()
