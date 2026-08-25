"""Point processing and intensity transforms for medical image enhancement."""

from __future__ import annotations

import cv2
import numpy as np


def image_negative(img: np.ndarray) -> np.ndarray:
    return 255 - img


def threshold_binary_inv(img: np.ndarray, value: int = 50) -> np.ndarray:
    _, out = cv2.threshold(img, value, 255, cv2.THRESH_BINARY_INV)
    return out


def grey_level_slicing(img: np.ndarray, lo: int = 100, hi: int = 200) -> np.ndarray:
    out = img.copy()
    mask = (img >= lo) & (img <= hi)
    out[mask] = 255
    return out


def contrast_stretch(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return img.copy()
    stretched = (img.astype(np.float64) - lo) * (255.0 / (hi - lo))
    return stretched.astype(np.uint8)


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(img)


def log_transform(img: np.ndarray, c: float = 1.5) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    y = c * np.log1p(x)
    y = y / y.max() if y.max() > 0 else y
    return (y * 255).astype(np.uint8)


def inverse_log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    x = img.astype(np.float32) / 255.0 + 1e-6
    y = c * (np.exp(x) - 1.0)
    y = y / y.max() if y.max() > 0 else y
    return (y * 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return (np.power(x, gamma) * 255).astype(np.uint8)


def bit_planes(img: np.ndarray) -> dict[str, np.ndarray]:
    planes = {"Original": img}
    titles = [
        "MSB (Bit 7)",
        "Bit 6",
        "Bit 5",
        "Bit 4",
        "Bit 3",
        "Bit 2",
        "Bit 1",
        "LSB (Bit 0)",
    ]
    for i, title in enumerate(reversed(titles)):
        plane = ((img >> i) & 1) * 255
        planes[title] = plane.astype(np.uint8)
    return planes


def unsharp_mask_pipeline(img: np.ndarray) -> dict[str, np.ndarray]:
    """Classic high-boost / edge-mask sharpening chain."""
    lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)
    sharpened_lap = cv2.add(img, lap_abs)

    sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))
    sobel_smooth = cv2.GaussianBlur(sobel, (5, 5), 1.2)

    mask = cv2.multiply(sobel, sobel_smooth, scale=1.0 / 255.0)
    final = cv2.add(img, mask)

    return {
        "Original": img,
        "Laplacian": lap_abs,
        "Original + Laplacian": sharpened_lap,
        "Sobel": sobel,
        "Smoothed Sobel": sobel_smooth,
        "Edge Mask": mask,
        "Final Sharpened": final,
    }


def run_enhancements(img: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        "negative": {"Original": img, "Negative": image_negative(img)},
        "threshold": {"Original": img, "Threshold INV (T=50)": threshold_binary_inv(img, 50)},
        "grey_level_slice": {"Original": img, "Grey Level Slice (100-200)": grey_level_slicing(img)},
        "contrast_stretch": {"Original": img, "Contrast Stretched": contrast_stretch(img)},
        "histogram_eq": {"Original": img, "Histogram Equalized": histogram_equalize(img)},
        "log": {"Original": img, "Log Transform": log_transform(img)},
        "inv_log": {"Original": img, "Inverse Log": inverse_log_transform(img)},
        "gamma": {
            "Original": img,
            "γ = 0.4": gamma_correct(img, 0.4),
            "γ = 1.0": gamma_correct(img, 1.0),
            "γ = 2.5": gamma_correct(img, 2.5),
        },
        "bit_planes": bit_planes(img),
        "unsharp_mask": unsharp_mask_pipeline(img),
    }
