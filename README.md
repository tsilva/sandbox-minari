<div align="center">
  <img src="logo.png" alt="sandbox-minari" width="512"/>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

  **🤖 Explore offline reinforcement learning datasets with Minari 📊**

</div>

## Overview

A sandbox for experimenting with [Minari](https://minari.farama.org/) and offline reinforcement learning. The repo focuses on small Python scripts that load datasets, train offline policies, and evaluate them in the recovered Gymnasium environment.

## 🚀 Quick Start

```bash
git clone https://github.com/tsilva/sandbox-minari.git
cd sandbox-minari
uv venv && source .venv/bin/activate
uv sync
```

## Scripts

- `quickstart.py` lists remote datasets, downloads one example dataset, and prints episode metadata.
- `main.py` trains a behavioral cloning baseline with `MLPClassifier` on MiniGrid FourRooms and evaluates it in the live environment.
- `train_cql.py` trains `DiscreteCQL` with `d3rlpy` on the same dataset for comparison.
- `train_breakout.py` trains a pixel-based behavioral cloning policy on Minari's Breakout expert dataset.

Run any script directly:

```bash
uv run python quickstart.py
uv run python main.py
uv run python train_cql.py
uv run python train_breakout.py --render-mode none
```

Use `--render-mode human` if you want to watch the Breakout policy play live.

## Dependencies

The project now installs Gymnasium's Atari extras so `atari/breakout/expert-v0` can recover `ALE/Breakout-v5`. Minari caches downloaded datasets under `~/.minari/`.

## 🔗 Resources

- [Minari Documentation](https://minari.farama.org/)
- [Minari GitHub](https://github.com/Farama-Foundation/Minari)
- [Gymnasium](https://gymnasium.farama.org/)

## License

[MIT](LICENSE)
