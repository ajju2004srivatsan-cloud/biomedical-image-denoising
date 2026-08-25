# Biomedical Image Denoising & Enhancement Study

**Author:** Ajay Srivatsan R (`CB.EN.U4ECE22005`)  
**Course:** 19ECE455 — Biomedical Signal Processing  
**Focus:** Spatial filtering for medical-image noise removal + point-processing enhancement

This repository packages a complete, runnable study: synthetic medical phantoms, noise models, classical OpenCV / scikit-image filters, quantitative PSNR comparisons, and saved output grids.

**Coursework artifacts included**
- Full assignment PDF: [`docs/19ECE455_A2_Biomedical_Signal_Processing.pdf`](docs/19ECE455_A2_Biomedical_Signal_Processing.pdf)
- Original figures from the report: [`data/assignment_figures/`](data/assignment_figures/) (noisy inputs + filter/enhancement result panels)

---

## What this study covers

### Part 1 — Noise removal (spatial filters)

| Noise type | Typical source | Best filter (PSNR vs clean) | Takeaway |
|---|---|---|---|
| **Rician** | Magnitude MRI | **Median** — 23.77 dB | Median edges out NLM on this phantom; NLM still strong on textured MRI |
| **Salt & pepper** | Sensor / transmission errors | **Median** — 32.44 dB | Clear winner for impulse noise |
| **Gaussian** | Thermal / electronic noise | **Median** — 29.92 dB | Bilateral close second (28.34 dB); plain Gaussian blur trails |
| **Periodic** | Interference / striping | **Bilateral** — 26.36 dB | Spatial filters help; FFT notch is better for strong striping |
| **Speckle** | Ultrasound / coherent imaging | **Hybrid Med+Bilateral** — 30.70 dB | Cascade filters beat single-stage Wiener here |

Full numbers are in [`outputs/metrics.json`](outputs/metrics.json). Comparison grids are under [`outputs/denoising/`](outputs/denoising/).

### Part 2 — Visual quality enhancement

- Image negative  
- Thresholding  
- Grey-level slicing  
- Contrast stretching  
- Histogram equalization  
- Log / inverse-log transforms  
- Gamma (power-law) correction  
- Bit-plane slicing  
- Laplacian / Sobel unsharp-mask sharpening chain  

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_study.py
```

Outputs land in:

```
data/phantoms/          # clean + noisy synthetic inputs
outputs/denoising/      # before/after filter comparison grids + best results
outputs/enhancement/    # point-processing result grids
outputs/metrics.json    # PSNR (dB) vs clean phantom for each filter
```

---

## Key inferences (from the study)

1. **Match the filter to the noise.** Median wins on salt-and-pepper; Gaussian blur alone is a poor default for diagnostic MRI.
2. **Non-Local Means** is the strongest general denoiser here when edges and fine tissue contrast matter, at the cost of runtime.
3. **Bilateral** preserves edges well on mild/moderate noise but needs tuning (`d`, `σ_color`, `σ_space`).
4. **Periodic interference** is only partly fixed by spatial filters — notch / FFT filtering is the proper next step for strong striping.
5. **Enhancement is complementary:** histogram equalization and γ&lt;1 improve dark-region visibility without claiming to “denoise.”
6. **Bit planes:** MSB (bit 7) carries most structural content; LSBs look like noise and are useful for compression or noise analysis.

---

## Project layout

```
docs/                                    # full assignment PDF
data/assignment_figures/                 # original PDF/report images
data/phantoms/                           # synthetic clean/noisy inputs
src/                                     # reproducible Python pipelines
outputs/denoising/  outputs/enhancement/ # regenerated comparison grids
```

---

## Notes

- Phantoms are **synthetic** so the study is fully reproducible without proprietary clinical data.
- Swap in your own grayscale PNG/JPG paths inside the pipelines if you want to re-run on assignment figures.
- Paper / portfolio use: cite OpenCV, NumPy, Matplotlib, and scikit-image.

---

## License

MIT — feel free to fork for coursework demos and portfolio showcases.
