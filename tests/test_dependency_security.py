from __future__ import annotations

import io
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import torch
from packaging.version import Version
from PIL import Image, ImageCms, UnidentifiedImageError

from train_breakout import preprocess_frame

ROOT = Path(__file__).resolve().parents[1]


def test_patched_dependency_versions_are_installed() -> None:
    assert Version(version("click")) >= Version("8.3.3")
    assert Version(version("pillow")) >= Version("12.3.0")
    assert Version(version("torch")) >= Version("2.13.0")
    assert Version(version("setuptools")) >= Version("83.0.0")
    assert Version(version("pygments")) >= Version("2.20.0")


def test_pillow_accepts_valid_frames_and_rejects_malformed_images() -> None:
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    processed = preprocess_frame(frame, 84)
    assert processed.shape == (84, 84)

    with pytest.raises((UnidentifiedImageError, OSError)):
        with Image.open(io.BytesIO(b"not an Atari frame")) as image:
            image.load()


def test_pillow_rejects_imagecms_output_mode_mismatch() -> None:
    profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(profile, profile, "RGBA", "RGBA")
    source = Image.new("RGBA", (8, 1), (1, 2, 3, 4))

    valid_output = Image.new("RGBA", source.size)
    assert transform.apply(source, valid_output).mode == "RGBA"

    mismatched_output = Image.new("L", source.size)
    with pytest.raises(ValueError, match="mode"):
        transform.apply(source, mismatched_output)


def test_torch_jit_and_gradient_smoke() -> None:
    layer = torch.nn.Linear(2, 1)
    scripted = torch.jit.script(layer)
    inputs = torch.tensor([[1.0, -1.0]], requires_grad=True)
    output = scripted(inputs)
    output.sum().backward()

    assert output.shape == (1, 1)
    assert inputs.grad is not None


def test_dependency_manifests_use_registry_sources_only() -> None:
    with (ROOT / "pyproject.toml").open("rb") as file:
        manifest = tomllib.load(file)
    with (ROOT / "uv.lock").open("rb") as file:
        lock = tomllib.load(file)

    declared = list(manifest["project"]["dependencies"])
    for dependencies in manifest["project"].get("optional-dependencies", {}).values():
        declared.extend(dependencies)
    assert not any(
        dependency.lower().startswith(("file:", "git:", "git+", "http:", "https:"))
        for dependency in declared
    )

    for package in lock["package"]:
        source = package.get("source", {})
        if package["name"] == "sandbox-minari" and set(source) & {
            "editable",
            "virtual",
            "directory",
        }:
            continue
        assert source == {"registry": "https://pypi.org/simple"}
