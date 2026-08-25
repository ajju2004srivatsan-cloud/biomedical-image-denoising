"""Synthetic medical phantoms and noise models for reproducible demos."""

from __future__ import annotations

import numpy as np

from utils import to_uint8


def brain_phantom(size: int = 256, seed: int = 7) -> np.ndarray:
    """Simple elliptical 'brain' phantom with soft tissue + ventricles."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    img = np.zeros((size, size), dtype=np.float64)

    # Outer skull / soft tissue
    skull = ((yy - cy) / (0.42 * size)) ** 2 + ((xx - cx) / (0.36 * size)) ** 2 <= 1
    img[skull] = 160

    # Gray matter ring
    gm = ((yy - cy) / (0.34 * size)) ** 2 + ((xx - cx) / (0.28 * size)) ** 2 <= 1
    img[gm] = 110

    # White matter core
    wm = ((yy - cy) / (0.22 * size)) ** 2 + ((xx - cx) / (0.18 * size)) ** 2 <= 1
    img[wm] = 180

    # Ventricles
    left = ((yy - cy) / (0.08 * size)) ** 2 + ((xx - (cx - 0.08 * size)) / (0.04 * size)) ** 2 <= 1
    right = ((yy - cy) / (0.08 * size)) ** 2 + ((xx - (cx + 0.08 * size)) / (0.04 * size)) ** 2 <= 1
    img[left | right] = 40

    # Mild anatomical texture
    rng = np.random.default_rng(seed)
    texture = rng.normal(0, 4, size=(size, size))
    img = np.clip(img + texture * skull, 0, 255)
    return to_uint8(img)


def bone_phantom(size: int = 256, seed: int = 11) -> np.ndarray:
    """Bone / cortical structure phantom for salt-and-pepper demos."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size), 30.0)
    cortex = ((yy - size * 0.5) / (0.3 * size)) ** 2 + ((xx - size * 0.5) / (0.45 * size)) ** 2 <= 1
    img[cortex] = 210
    marrow = ((yy - size * 0.5) / (0.18 * size)) ** 2 + ((xx - size * 0.5) / (0.28 * size)) ** 2 <= 1
    img[marrow] = 90
    rng = np.random.default_rng(seed)
    img += rng.normal(0, 3, size=(size, size))
    return to_uint8(img)


def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0, sigma, image.shape)
    return to_uint8(noisy)


def add_salt_pepper(image: np.ndarray, amount: float = 0.05, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    n = out.size
    n_salt = int(amount * n * 0.5)
    n_pepper = int(amount * n * 0.5)
    coords = rng.choice(n, n_salt + n_pepper, replace=False)
    flat = out.ravel()
    flat[coords[:n_salt]] = 255
    flat[coords[n_salt:]] = 0
    return out


def add_speckle(image: np.ndarray, variance: float = 0.04, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(variance), image.shape)
    noisy = image.astype(np.float64) * (1.0 + noise)
    return to_uint8(noisy)


def add_rician(image: np.ndarray, sigma: float = 18.0, seed: int = 4) -> np.ndarray:
    """Rician noise typical of magnitude MRI."""
    rng = np.random.default_rng(seed)
    real = image.astype(np.float64) + rng.normal(0, sigma, image.shape)
    imag = rng.normal(0, sigma, image.shape)
    return to_uint8(np.sqrt(real**2 + imag**2))


def add_periodic(image: np.ndarray, amplitude: float = 35.0, freq: float = 0.08) -> np.ndarray:
    """Sinusoidal interference (stripe / interference pattern)."""
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    pattern = amplitude * np.sin(2 * np.pi * freq * xx + 0.7 * np.sin(2 * np.pi * 0.03 * yy))
    return to_uint8(image.astype(np.float64) + pattern)


def build_dataset(size: int = 256) -> dict[str, dict[str, np.ndarray]]:
    brain = brain_phantom(size)
    bone = bone_phantom(size)
    return {
        "rician": {"clean": brain, "noisy": add_rician(brain)},
        "salt_pepper": {"clean": bone, "noisy": add_salt_pepper(bone)},
        "gaussian": {"clean": brain, "noisy": add_gaussian_noise(brain)},
        "periodic": {"clean": brain, "noisy": add_periodic(brain)},
        "speckle": {"clean": brain, "noisy": add_speckle(brain)},
        "enhancement": {"clean": brain, "noisy": brain},
    }
