"""
MedImg Denoise — end-to-end medical image denoising & enhancement runner.

Run from repo root:
  python src/run_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from denoise import PIPELINES
from enhance import run_enhancements
from phantoms import build_dataset
from utils import DATA_DIR, OUTPUT_DIR, ensure_dirs, psnr, save_comparison, save_image


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = DATA_DIR / "samples"
GALLERY = ROOT / "gallery"
GALLERY_DENOISE = GALLERY / "denoising"
GALLERY_ENHANCE = GALLERY / "enhancement"

SAMPLE_FILES = {
    "rician": SAMPLES / "mri_rician.png",
    "salt_pepper": SAMPLES / "bone_salt_pepper.png",
    "gaussian": SAMPLES / "brain_gaussian.png",
    "periodic": SAMPLES / "brain_periodic.png",
    "speckle": SAMPLES / "brain_speckle.png",
}

TITLES = {
    "rician": "Rician Noise Denoising (MRI)",
    "salt_pepper": "Salt & Pepper Denoising (Bone)",
    "gaussian": "Gaussian Noise Denoising (Brain MRI)",
    "periodic": "Periodic Noise Removal",
    "speckle": "Speckle Noise Denoising",
}

PREFERRED = {
    "rician": "Non-Local Means",
    "salt_pepper": "Median",
    "gaussian": "NLM (h=10)",
    "periodic": "Bilateral (d=15)",
    "speckle": "Hybrid (Med+Bilat)",
}


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def main() -> None:
    phantom_dir = DATA_DIR / "phantoms"
    ensure_dirs(phantom_dir, GALLERY_DENOISE, GALLERY_ENHANCE, OUTPUT_DIR / "denoising", OUTPUT_DIR / "enhancement")

    dataset = build_dataset(size=256)
    metrics: dict[str, dict[str, float]] = {}

    for noise_type, pipeline in PIPELINES.items():
        sample_path = SAMPLE_FILES.get(noise_type)
        if sample_path and sample_path.exists():
            noisy = load_gray(sample_path)
            clean = dataset[noise_type]["clean"]  # phantom GT only for synthetic branch
            use_real = True
        else:
            clean = dataset[noise_type]["clean"]
            noisy = dataset[noise_type]["noisy"]
            use_real = False
            save_image(phantom_dir / f"{noise_type}_clean.png", clean)
            save_image(phantom_dir / f"{noise_type}_noisy.png", noisy)

        results = pipeline(noisy)
        metrics[noise_type] = {}
        ref = clean if not use_real else noisy
        for name, img in results.items():
            if "original" in name.lower():
                continue
            metrics[noise_type][name] = round(psnr(ref, img), 2)

        best_name = PREFERRED.get(noise_type)
        if best_name and best_name in results:
            save_image(GALLERY_DENOISE / f"{noise_type}_best.png", results[best_name])

        out = GALLERY_DENOISE / f"{noise_type}.png"
        save_comparison(
            results,
            out,
            TITLES[noise_type],
            ncols=3 if len(results) > 4 else len(results),
            show_psnr_vs=ref if not use_real else None,
        )
        # keep legacy outputs/ mirror for compatibility
        save_comparison(
            results,
            OUTPUT_DIR / "denoising" / f"{noise_type}_comparison.png",
            TITLES[noise_type],
            ncols=3 if len(results) > 4 else len(results),
            show_psnr_vs=ref if not use_real else None,
        )
        print(f"[denoise] {noise_type} ({'sample' if use_real else 'phantom'})")

    # enhancement on a lightly cleaned MRI sample when available
    if SAMPLE_FILES["rician"].exists():
        base = cv2.medianBlur(load_gray(SAMPLE_FILES["rician"]), 5)
    else:
        base = dataset["enhancement"]["clean"]
        save_image(phantom_dir / "enhancement_base.png", base)

    enhancements = run_enhancements(base)
    for name, images in enhancements.items():
        save_comparison(
            images,
            GALLERY_ENHANCE / f"{name}.png",
            f"Enhancement — {name.replace('_', ' ').title()}",
            ncols=3 if len(images) > 4 else min(len(images), 4),
        )
        save_comparison(
            images,
            OUTPUT_DIR / "enhancement" / f"{name}.png",
            f"Enhancement — {name.replace('_', ' ').title()}",
            ncols=3 if len(images) > 4 else min(len(images), 4),
        )
        print(f"[enhance] {name}")

    (GALLERY / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nGallery → {GALLERY}")
    print(f"Metrics → {GALLERY / 'metrics.json'}")


if __name__ == "__main__":
    main()
