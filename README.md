<div align="center">
  <img src="logo.png" alt="sandbox-minari" width="512"/>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

  **🤖 Explore offline reinforcement learning datasets with Minari 📊**

</div>

## Overview

A sandbox for experimenting with [Minari](https://minari.farama.org/) — a Python API and dataset repository for offline reinforcement learning. Load, inspect, and learn from pre-collected environment interaction data without needing to train agents from scratch.

## 🚀 Quick Start

```bash
git clone https://github.com/tsilva/sandbox-minari.git
cd sandbox-minari
uv venv && source .venv/bin/activate
uv pip install -e .
```

## 📦 Dependencies

Install Minari and common RL libraries:

```bash
uv pip install minari gymnasium
```

## 🧪 Usage

```python
import minari

# List available datasets
datasets = minari.list_remote_datasets()
print(datasets)

# Download and load a dataset
dataset = minari.load_dataset("CartPole-v1-test-v0", download=True)

# Iterate through episodes
for episode in dataset.iterate_episodes():
    print(f"Episode length: {episode.total_timesteps}")
```

## 🔗 Resources

- [Minari Documentation](https://minari.farama.org/)
- [Minari GitHub](https://github.com/Farama-Foundation/Minari)
- [Gymnasium](https://gymnasium.farama.org/)

## License

[MIT](LICENSE)
