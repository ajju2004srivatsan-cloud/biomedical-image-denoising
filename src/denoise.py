"""Spatial / edge-preserving filters for common medical-image noise types."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.restoration import wiener
from skimage.util import img_as_float, img_as_ubyte


def median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def gaussian(img: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def nlm(img: np.ndarray, h: float = 12, template: int = 7, search: int = 21) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=template, searchWindowSize=search)


def mean(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.blur(img, (ksize, ksize))


def wiener_filter(img: np.ndarray, ksize: int = 5, balance: float = 0.3) -> np.ndarray:
    psf = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    restored = wiener(img_as_float(img), psf, balance=balance)
    return img_as_ubyte(np.clip(restored, 0, 1))


def gaussian_sharpen(img: np.ndarray) -> np.ndarray:
    blurred = gaussian(img, 5, 2.0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(blurred, -1, kernel)


def mean_median_combo(img: np.ndarray) -> np.ndarray:
    return median(mean(img, 5), 3)


def hybrid_median_bilateral(img: np.ndarray) -> np.ndarray:
    return bilateral(median(img, 3), 9, 75, 75)


def mean_shift(img: np.ndarray, sp: int = 15, sr: int = 30) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    filtered = cv2.pyrMeanShiftFiltering(bgr, sp, sr)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)


def denoise_rician(noisy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Original (noisy)": noisy,
        "Non-Local Means": nlm(noisy, h=12, template=9, search=25),
        "Gaussian": gaussian(noisy, 7, 2.5),
        "Median": median(noisy, 5),
    }


def denoise_salt_pepper(noisy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Original (noisy)": noisy,
        "Median": median(noisy, 5),
        "Gaussian": gaussian(noisy, 5, 1.5),
        "Bilateral": bilateral(noisy),
        "Non-Local Means": nlm(noisy, h=15),
        "Wiener": wiener_filter(noisy),
    }


def denoise_gaussian(noisy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Original (noisy)": noisy,
        "Gaussian (5x5)": gaussian(noisy, 5, 1.5),
        "Median (5x5)": median(noisy, 5),
        "Bilateral (d=9)": bilateral(noisy),
        "NLM (h=10)": nlm(noisy, h=10),
        "Wiener (5x5)": wiener_filter(noisy, balance=0.3),
    }


def denoise_periodic(noisy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Original (noisy)": noisy,
        "Median (7x7)": median(noisy, 7),
        "Bilateral (d=15)": bilateral(noisy, d=15),
        "Gaussian + Sharpen": gaussian_sharpen(noisy),
        "Mean + Median": mean_median_combo(noisy),
    }


def denoise_speckle(noisy: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Original (noisy)": noisy,
        "Median (5x5)": median(noisy, 5),
        "Hybrid (Med+Bilat)": hybrid_median_bilateral(noisy),
        "Wiener": wiener_filter(noisy, ksize=3, balance=0.25),
        "Mean Shift": mean_shift(noisy),
    }


PIPELINES = {
    "rician": denoise_rician,
    "salt_pepper": denoise_salt_pepper,
    "gaussian": denoise_gaussian,
    "periodic": denoise_periodic,
    "speckle": denoise_speckle,
}
