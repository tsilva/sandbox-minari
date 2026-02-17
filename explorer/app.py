"""Minari Dataset Explorer — FastAPI web UI for browsing offline RL datasets."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import minari
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Minari Explorer")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_dataset_cache: dict[str, minari.MinariDataset] = {}


@lru_cache(maxsize=1)
def get_remote_catalog() -> dict[str, dict]:
    """Fetch the remote dataset catalog (cached after first call)."""
    return minari.list_remote_datasets(latest_version=True)


def load_dataset_cached(dataset_id: str) -> minari.MinariDataset:
    """Load (and download if needed) a dataset, caching the result."""
    if dataset_id not in _dataset_cache:
        _dataset_cache[dataset_id] = minari.load_dataset(dataset_id, download=True)
    return _dataset_cache[dataset_id]


def compute_dataset_stats(dataset: minari.MinariDataset) -> dict[str, Any]:
    """Iterate episodes once and compute stats + Plotly chart HTML."""
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    action_counts: dict[Any, int] = {}
    episodes_info: list[dict] = []

    for ep in dataset:
        total_reward = float(np.sum(ep.rewards))
        length = len(ep.rewards)
        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        terminated = bool(ep.terminations[-1]) if len(ep.terminations) > 0 else False
        truncated = bool(ep.truncations[-1]) if len(ep.truncations) > 0 else False

        episodes_info.append({
            "index": ep.id,
            "length": length,
            "reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        for a in np.asarray(ep.actions).flat:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1

    charts = []

    # Reward distribution
    if episode_rewards:
        fig = go.Figure(go.Histogram(x=episode_rewards, nbinsx=30))
        fig.update_layout(title="Reward Distribution", xaxis_title="Total Reward", yaxis_title="Count", margin=dict(t=40, b=30, l=40, r=20), height=300)
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Episode length distribution
    if episode_lengths:
        fig = go.Figure(go.Histogram(x=episode_lengths, nbinsx=30))
        fig.update_layout(title="Episode Lengths", xaxis_title="Steps", yaxis_title="Count", margin=dict(t=40, b=30, l=40, r=20), height=300)
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Action frequency
    if action_counts:
        sorted_actions = sorted(action_counts.items())
        fig = go.Figure(go.Bar(x=[str(a) for a, _ in sorted_actions], y=[c for _, c in sorted_actions]))
        fig.update_layout(title="Action Frequency", xaxis_title="Action", yaxis_title="Count", margin=dict(t=40, b=30, l=40, r=20), height=300)
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    return {"episodes": episodes_info, "charts": charts}


def _obs_at_step(observations: Any, step: int) -> Any:
    """Extract the observation at a given step, handling dict/ndarray."""
    if isinstance(observations, dict):
        return {k: _obs_at_step(v, step) for k, v in observations.items()}
    return observations[step]


def render_observation(obs: Any) -> str:
    """Render an observation value as HTML (image or formatted text)."""
    if isinstance(obs, dict):
        parts = []
        for k, v in obs.items():
            parts.append(f"<strong>{k}:</strong> {render_observation(v)}")
        return "<br>".join(parts)

    arr = np.asarray(obs)

    # Image-like: 2D or 3D array with reasonable dimensions
    if arr.ndim >= 2 and arr.shape[0] <= 256 and arr.shape[1] <= 256:
        from PIL import Image
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                img = Image.fromarray(arr[:, :, 0].astype(np.uint8), mode="L")
            else:
                mode = "RGB" if arr.shape[2] == 3 else "RGBA"
                img = Image.fromarray(arr.astype(np.uint8), mode=mode)
        else:
            return f"<code>{arr}</code>"

        # Scale up small images for visibility
        scale = max(1, 280 // max(img.width, img.height))
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{b64}" class="obs-image">'

    # Scalar or small flat array
    if arr.ndim == 0:
        return f"<code>{arr.item()}</code>"
    if arr.size <= 20:
        return f"<code>{arr.tolist()}</code>"
    return f"<code>shape={arr.shape}, dtype={arr.dtype}</code>"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def catalog(request: Request):
    datasets = get_remote_catalog()
    return templates.TemplateResponse("catalog.html", {"request": request, "datasets": datasets})


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = ""):
    datasets = get_remote_catalog()
    query = q.lower()
    filtered = {k: v for k, v in datasets.items() if query in k.lower()}
    rows = []
    for ds_id, info in filtered.items():
        rows.append(
            f'<tr><td><a href="/dataset/{ds_id}">{ds_id}</a></td>'
            f'<td>{info.get("total_episodes", "—")}</td>'
            f'<td>{info.get("total_steps", "—")}</td></tr>'
        )
    return HTMLResponse("\n".join(rows))


@app.get("/dataset/{dataset_id:path}/episode/{ep_idx:int}/step/{step:int}", response_class=HTMLResponse)
async def step_fragment(request: Request, dataset_id: str, ep_idx: int, step: int):
    dataset = load_dataset_cached(dataset_id)
    episode = dataset[ep_idx]
    total_steps = len(episode.rewards)
    step = max(0, min(step, total_steps - 1))

    obs = _obs_at_step(episode.observations, step)
    obs_html = render_observation(obs)
    reward = float(episode.rewards[step])
    cumulative_reward = float(np.sum(episode.rewards[: step + 1]))
    action = episode.actions[step]
    if isinstance(action, np.ndarray):
        action = action.tolist()

    return templates.TemplateResponse("step_fragment.html", {
        "request": request,
        "dataset_id": dataset_id,
        "ep_idx": ep_idx,
        "step": step,
        "total_steps": total_steps,
        "obs_html": obs_html,
        "action": action,
        "reward": reward,
        "cumulative_reward": cumulative_reward,
    })


@app.get("/dataset/{dataset_id:path}/episode/{ep_idx:int}", response_class=HTMLResponse)
async def episode_view(request: Request, dataset_id: str, ep_idx: int):
    dataset = load_dataset_cached(dataset_id)
    episode = dataset[ep_idx]
    total_steps = len(episode.rewards)
    total_reward = float(np.sum(episode.rewards))

    step = 0
    obs = _obs_at_step(episode.observations, step)
    obs_html = render_observation(obs)
    reward = float(episode.rewards[step])
    cumulative_reward = reward
    action = episode.actions[step]
    if isinstance(action, np.ndarray):
        action = action.tolist()

    return templates.TemplateResponse("episode.html", {
        "request": request,
        "dataset_id": dataset_id,
        "ep_idx": ep_idx,
        "step": step,
        "total_steps": total_steps,
        "total_reward": total_reward,
        "obs_html": obs_html,
        "action": action,
        "reward": reward,
        "cumulative_reward": cumulative_reward,
    })


@app.get("/dataset/{dataset_id:path}", response_class=HTMLResponse)
async def dataset_detail(request: Request, dataset_id: str):
    dataset = load_dataset_cached(dataset_id)
    spec = dataset.spec
    stats = compute_dataset_stats(dataset)

    env_spec = None
    if spec.env_spec is not None:
        env_spec = spec.env_spec.id

    return templates.TemplateResponse("dataset.html", {
        "request": request,
        "dataset_id": dataset_id,
        "env_spec": env_spec,
        "total_episodes": spec.total_episodes,
        "total_steps": spec.total_steps,
        "obs_space": str(spec.observation_space),
        "act_space": str(spec.action_space),
        "charts": stats["charts"],
        "episodes": stats["episodes"],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
