# CLAUDE.md

## Project overview

Sandbox for exploring offline reinforcement learning with [Minari](https://minari.farama.org/). Contains example pipelines that download pre-collected RL datasets, train agents offline, and evaluate them in environments.

## Development setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Requires Python 3.12+. Uses `uv` as the package manager.

## Scripts

- `quickstart.py` — Demonstrates Minari's core API: listing remote datasets, loading one, inspecting metadata, and iterating episodes.
- `main.py` — Full pipeline: downloads MiniGrid FourRooms dataset, trains an MLPClassifier (behavioral cloning) with early stopping, evaluates in the live environment with pygame rendering.
- `train_cql.py` — Same dataset but trains DiscreteCQL via d3rlpy for comparison with behavioral cloning.

Run any script directly:

```bash
python quickstart.py
python main.py
python train_cql.py
```

## Key dependencies

- `minari` — Offline RL dataset API (with HuggingFace, HDF5, and create extras)
- `gymnasium` / `minigrid` — RL environments
- `scikit-learn` — MLPClassifier for behavioral cloning
- `d3rlpy` — Offline RL algorithms (CQL)
- `numpy`, `Pillow`, `tqdm` — Data processing and progress display

## Conventions

- Pre-commit hook runs gitleaks for secret detection
- All scripts follow a 4-step pattern: load dataset, prepare data, train, evaluate
- Dataset downloads are cached locally by Minari (~/.minari/)
- Evaluation renders with `render_mode="human"` (opens pygame window)

## Important

- Keep README.md up to date with any significant project changes
