"""Shared helpers for loading images, metrics, and saving comparison grids."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher is better)."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    mse = np.mean((ref - est) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def save_comparison(
    images: dict[str, np.ndarray],
    out_path: Path,
    title: str,
    ncols: int | None = None,
    show_psnr_vs: np.ndarray | None = None,
) -> None:
    """Save a labeled grid of grayscale images."""
    n = len(images)
    ncols = ncols or min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        label = name
        if show_psnr_vs is not None and name.lower() not in {"original", "clean", "noisy"}:
            label = f"{name}\nPSNR: {psnr(show_psnr_vs, img):.2f} dB"
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), to_uint8(image))
