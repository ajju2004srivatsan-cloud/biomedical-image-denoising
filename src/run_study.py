"""
Biomedical Image Denoising & Enhancement Study
----------------------------------------------
Reproducible pipeline for Assignment-02 style experiments:
  1) Spatial filters for Rician / salt-pepper / Gaussian / periodic / speckle noise
  2) Point processing & intensity transforms for visual quality improvement

Run from repo root:
  python src/run_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python src/run_study.py` without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from denoise import PIPELINES
from enhance import run_enhancements
from phantoms import build_dataset
from utils import DATA_DIR, OUTPUT_DIR, ensure_dirs, psnr, save_comparison, save_image


TITLES = {
    "rician": "Rician Noise — MRI Magnitude Denoising",
    "salt_pepper": "Salt & Pepper Noise — Bone Close-up Denoising",
    "gaussian": "Gaussian Noise — Brain Scan Denoising",
    "periodic": "Periodic Noise — Interference Pattern Removal",
    "speckle": "Speckle Noise — Ultrasound-like Denoising",
}


def main() -> None:
    phantom_dir = DATA_DIR / "phantoms"
    denoise_out = OUTPUT_DIR / "denoising"
    enhance_out = OUTPUT_DIR / "enhancement"
    ensure_dirs(phantom_dir, denoise_out, enhance_out)

    dataset = build_dataset(size=256)
    metrics: dict[str, dict[str, float]] = {}

    # --- Part 1: noise removal ---
    for noise_type, pipeline in PIPELINES.items():
        clean = dataset[noise_type]["clean"]
        noisy = dataset[noise_type]["noisy"]
        save_image(phantom_dir / f"{noise_type}_clean.png", clean)
        save_image(phantom_dir / f"{noise_type}_noisy.png", noisy)

        results = pipeline(noisy)
        # Keep clean reference for fair PSNR vs ground truth
        metrics[noise_type] = {}
        for name, img in results.items():
            if "original" in name.lower():
                continue
            metrics[noise_type][name] = round(psnr(clean, img), 2)

        # Rank best filter by PSNR vs clean phantom
        ranked = sorted(metrics[noise_type].items(), key=lambda kv: kv[1], reverse=True)
        best_name = ranked[0][0] if ranked else None
        if best_name:
            save_image(denoise_out / f"{noise_type}_best.png", results[best_name])

        save_comparison(
            results,
            denoise_out / f"{noise_type}_comparison.png",
            TITLES[noise_type],
            ncols=3 if len(results) > 4 else len(results),
            show_psnr_vs=clean,
        )
        print(f"[denoise] {noise_type}: best = {best_name} ({ranked[0][1]} dB)" if ranked else f"[denoise] {noise_type}")

    # --- Part 2: enhancement ---
    base = dataset["enhancement"]["clean"]
    save_image(phantom_dir / "enhancement_base.png", base)
    enhancements = run_enhancements(base)
    for name, images in enhancements.items():
        ncols = 3 if len(images) > 4 else min(len(images), 4)
        save_comparison(
            images,
            enhance_out / f"{name}.png",
            f"Enhancement — {name.replace('_', ' ').title()}",
            ncols=ncols,
        )
        print(f"[enhance] {name}")

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote metrics → {metrics_path}")
    print(f"Comparison grids → {denoise_out}")
    print(f"Enhancement grids → {enhance_out}")


if __name__ == "__main__":
    main()
