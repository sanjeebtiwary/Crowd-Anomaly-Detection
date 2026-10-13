# Crowd Anomaly Detection

**Flag unusual events in pedestrian-scene surveillance video by learning what "normal" motion looks like.**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/sanjeebtiwary/Crowd-Anomaly-Detection)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Optical%20Flow-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GMM-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-14%20passing-2EA44F?style=for-the-badge)

---

## Contents

[Overview](#overview) · [How it works](#how-it-works) · [Quick start](#quick-start) · [Dataset](#dataset) · [Usage](#usage) · [Configuration](#configuration) · [Project structure](#project-structure) · [Testing](#testing) · [What changed](#what-changed-from-the-original-notebook) · [Limitations](#limitations) · [Roadmap](#roadmap) · [References](#references) · [License](#license)

## Overview

Abnormal events in crowds can't be enumerated in advance, so this project uses a **one-class** approach: fit a model on *normal-only* footage, then score how far new observations deviate from it. In the UCSD data, anomalies include **bikers, skaters, carts and wheelchairs** on a pedestrian walkway, and unusual pedestrian motion.

It provides:

- a reusable Python package (`crowd_anomaly/`) and a command line (`train`, `evaluate`, `predict`, `demo`)
- frame-level evaluation (ROC-AUC, EER, precision/recall) with plots
- anomaly **localization overlays**: annotated output video highlights the cells that look unusual
- models that accept **any video** (`.avi`, `.mp4`, ...) or folder of frames, resized automatically
- a test suite and a synthetic-data demo that runs without downloading anything

> **Results.** No UCSD accuracy figures are published here yet. Run `train --evaluate` on your copy of the data and report your own numbers (see [Roadmap](#roadmap)). The bundled demo uses synthetic data and says nothing about UCSD performance.

## How it works

```mermaid
flowchart LR
    A[Video / frames] --> B[Grayscale, resize, blur]
    B --> C[Dense optical flow<br/>Farneback]
    C --> D[Pool on grid:<br/>mean dx, dy, magnitude per cell]
    D --> E[Gaussian mixture fit on<br/>NORMAL training cells]
    E --> F[Cell score =<br/>negative log-likelihood]
    F --> G[Frame score =<br/>mean of top-k cells,<br/>temporally smoothed]
    G --> H{score > threshold?}
    H -- yes --> I[Flag frame + highlight cells]
    H -- no --> J[Normal]
```

1. **Features.** For each pair of consecutive frames, dense Farneback optical flow is pooled on a `6 x 8` grid. Each cell keeps `[mean dx, mean dy, mean magnitude]`.
2. **Normality model.** Every cell of every training frame is a sample `[dx, dy, magnitude, y, x]`. Position is included because motion near the camera looks different from motion far away (perspective). A `StandardScaler` and a `GaussianMixture` are fitted on these samples.
3. **Scoring.** A cell's anomaly score is its negative log-likelihood (higher = more unusual). The frame score is the mean of the `top_k` highest cell scores, then smoothed over time, so a single moving object is not drowned out by a mostly static scene.
4. **Thresholding.** Whole training clips are held out (never individual frames) and the threshold is the 99th percentile of their scores, i.e. it targets roughly a 1% false-alarm rate on normal footage.

> **Naming.** The original notebook called this "MDT". The papers' Mixture of Dynamic Textures is a mixture of linear dynamical systems over appearance and dynamics. This project implements a simpler **optical-flow + Gaussian mixture** model that follows the same "learn normal, score deviation" principle but is *not* a reimplementation of the papers' MDT.

## Quick start

```bash
git clone https://github.com/sanjeebtiwary/Crowd-Anomaly-Detection.git
cd Crowd-Anomaly-Detection

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m crowd_anomaly demo         # ~10 s smoke test on synthetic data, no download needed
```

The demo trains on a tiny synthetic dataset, evaluates, and writes metrics, plots and an annotated video to a temp folder (path is printed).

## Dataset

**[UCSD Anomaly Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset/data)** - a stationary, elevated camera over pedestrian walkways. Training clips are normal; test clips contain anomalies.

| Subset | Scene | Train clips | Test clips | Frame size |
|---|---|---:|---:|---|
| **Ped1** | People walking toward/away from the camera (perspective distortion) | 34 | 36 | 238 x 158 |
| **Ped2** | Movement parallel to the camera plane | 16 | 12 | 360 x 240 |

Clips are ~200 frames of `.tif` images. Ground truth: a frame-level flag for every test clip (the `UCSDpedX.m` file) and pixel masks (`TestXXX_gt/`) for a subset.

Download and extract it anywhere. `--data-root` may point at the download folder, `UCSD_Anomaly_Dataset`, or the subset folder itself; the code finds `UCSDped1` / `UCSDped2` automatically:

```text
UCSD_Anomaly_Dataset/
├── UCSDped1/
│   ├── Train/Train001/001.tif ...        Train034/
│   └── Test/Test001/ ... Test036/        (+ Test0XX_gt/ masks, + UCSDped1.m)
└── UCSDped2/ ...
```

<table>
  <tr>
    <td align="center" width="50%"><strong>Cart</strong><br><img src="https://github.com/priyanka011011/Crowd-Anomaly-Detection/assets/63203112/92aff7f9-a790-472d-b7f9-4211e4cf9147" alt="Cart anomaly in Ped1" width="100%"></td>
    <td align="center" width="50%"><strong>Wheelchair</strong><br><img src="https://github.com/priyanka011011/Crowd-Anomaly-Detection/assets/63203112/fe32f054-303d-44cb-bf29-2e7d6791beb9" alt="Wheelchair anomaly in Ped1" width="100%"></td>
  </tr>
  <tr>
    <td align="center" width="50%"><strong>Skater</strong><br><img src="https://github.com/priyanka011011/Crowd-Anomaly-Detection/assets/63203112/5c060288-5138-43a0-9d24-787c8531a061" alt="Skater anomaly in Ped1" width="100%"></td>
    <td align="center" width="50%"><strong>Biker</strong><br><img src="https://github.com/priyanka011011/Crowd-Anomaly-Detection/assets/63203112/785b87bc-5a75-42ff-baad-4e949e10269f" alt="Biker anomaly in Ped1" width="100%"></td>
  </tr>
</table>

## Usage

### 1. Train (and optionally evaluate)

```bash
python -m crowd_anomaly train \
    --data-root /path/to/UCSD_Anomaly_Dataset --subset ped1 \
    --model-out models/ped1.joblib --evaluate --jobs -1
```

Useful options: `--components 12`, `--auto-components` (choose by BIC), `--grid 8 12`, `--top-k 5`, `--val-fraction 0.2`, `--max-clips 10` (quick experiments), `--seed 0`.

### 2. Evaluate

```bash
python -m crowd_anomaly evaluate --data-root /path/to/UCSD_Anomaly_Dataset \
    --subset ped1 --model models/ped1.joblib --report-dir reports/ped1
```

Writes `metrics.json` (AUC, EER, average precision, precision/recall/F1 at the calibrated threshold, per-clip AUC), `roc.png` and `timelines.png`. AUC and EER are computed over frames pooled across all labelled test clips, the usual UCSD frame-level protocol.

### 3. Predict on a video or folder of frames

```bash
python -m crowd_anomaly predict --source test.avi --model models/ped1.joblib --out outputs/
# or the original entry point, which defaults to test.avi + models/model.joblib:
python load_predict.py
```

Outputs: `scores.csv` (`frame, score, threshold, is_anomaly`), `scores.png` (score timeline with flagged segments shaded), and `annotated.mp4` (red border and label on flagged frames, red tint on unusual cells). The console lists anomalous frame ranges; segments shorter than `--min-segment` frames (default 3) are not reported.

### 4. Use it from Python

```python
from crowd_anomaly import Config, CrowdAnomalyModel
from crowd_anomaly.data import discover_clips, find_subset_dir
from crowd_anomaly.features import extract_many, clip_features

sub = find_subset_dir("UCSD_Anomaly_Dataset", "ped1")
feats, size = extract_many(discover_clips(sub, "Train"), Config(), n_jobs=-1)
model = CrowdAnomalyModel(Config(n_components=8)).fit(feats, size)
model.save("models/ped1.joblib")

pred = CrowdAnomalyModel.load("models/ped1.joblib").predict(
    clip_features("test.avi", model.cfg, model.frame_size))
pred["scores"], pred["flags"], pred["cell_scores"]
```

The notebook `model.ipynb` walks through the same steps interactively. Without the dataset it falls back to synthetic data so every cell runs.

## Configuration

Everything lives in one `Config` dataclass, saved inside the model file so inference always uses the training settings.

| Setting | Default | Effect |
|---|---|---|
| `grid_rows`, `grid_cols` | 6, 8 | Spatial pooling resolution; finer grids localize better but are noisier |
| `n_components` / `auto_components` | 8 / off | Mixture size, fixed or chosen by BIC from `component_candidates` |
| `covariance_type`, `reg_covar` | `full`, 1e-4 | GMM covariance structure and regularisation |
| `top_k` | 3 | Number of worst cells averaged into the frame score |
| `smooth_sigma` | 2.0 | Temporal Gaussian smoothing of the frame score (0 = off) |
| `val_fraction` | 0.2 | Share of training *clips* held out for thresholds |
| `threshold_percentile` | 99.0 | Higher = fewer false alarms, lower recall |
| `frame_size` | auto | Working resolution `(w, h)`; inferred from the first training frame |
| `blur_ksize`, `flow_*` | 3, Farneback defaults | Preprocessing and optical-flow parameters |
| `seed` | 0 | Reproducibility |

## Project structure

```text
Crowd-Anomaly-Detection/
├── crowd_anomaly/
│   ├── config.py        # Config dataclass + validation
│   ├── data.py          # dataset discovery, frame/video reading, ground-truth parsing
│   ├── features.py      # optical flow + grid pooling, parallel extraction
│   ├── model.py         # CrowdAnomalyModel: fit / score / threshold / save / load
│   ├── evaluate.py      # AUC, EER, precision/recall, plots
│   ├── inference.py     # predict on video/folder; CSV, plot, annotated video
│   ├── synthetic.py     # tiny synthetic UCSD-style dataset (tests and demo)
│   └── cli.py           # python -m crowd_anomaly {train,evaluate,predict,demo}
├── tests/test_pipeline.py
├── load_predict.py      # backward-compatible entry point
├── model.ipynb          # interactive walk-through
├── requirements.txt / requirements-dev.txt
└── README.md
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The 14 tests cover dataset discovery, ground-truth parsing (`.m` ranges and pixel masks), feature shapes, helper functions, model save/load round-trips, end-to-end detection on synthetic data, video input at a different resolution, `--auto-components`, and clean CLI error handling. They run on synthetic data only; they verify the code works, not how accurate it is on real footage.

## What changed from the original notebook

| Original | Now |
|---|---|
| `mdt_model.save('model.h5')` crashed (scikit-learn GMM has no `.save()`) | `joblib` persistence with version check, config, thresholds and frame size stored |
| Hard-coded `D:\...` paths; 35 directories typed by hand (`Train020` twice) | `--data-root` plus automatic discovery |
| Frames from all clips concatenated, so flow was computed across clip boundaries | Features computed per clip |
| `extract_features` defined twice; first version collapsed flow to 2 numbers | One grid-pooled feature (dx, dy, magnitude per cell) |
| Flattened full-resolution flow (~75k dims) fed to a GMM | 144 numbers per frame pair, modelled per cell (5-D) |
| No threshold, no evaluation | Held-out-clip threshold; AUC, EER, precision/recall, plots |
| Frame-level only, no localization | Cell scores drive an overlay on unusual regions |
| Notebook only | Package, CLI, tests, demo |
| `random` state not fixed | Seeded and reproducible |

## Limitations

- The model is **scene-specific**: a new camera, angle or lighting generally needs retraining.
- A coarse optical-flow grid can miss small, slow or non-moving anomalies (e.g. a stationary abandoned object).
- Localization is cell-level (default 6 x 8), not pixel-accurate, and is not yet evaluated against the pixel masks.
- A high score means "unusual compared with training", not "dangerous". Keep a human in the loop and consider privacy and retention rules for any real deployment.
- Real-time throughput was not benchmarked. Training was verified on Python 3.12 with synthetic data; run `train --evaluate` on the real dataset to see actual accuracy.
- `joblib` files are pickles: only load models you trained or trust.

## Roadmap

- [ ] Publish a results table for Ped1 and Ped2 (AUC / EER)
- [ ] Pixel-level localization evaluation against the `*_gt` masks
- [ ] True Mixture of Dynamic Textures baseline for comparison
- [ ] Richer features (flow histograms, temporal stacking of neighbouring frames)
- [ ] Streaming/real-time mode with a causal (non-smoothed) score

## References

1. V. Mahadevan, W. Li, V. Bhalodia, N. Vasconcelos. "Anomaly Detection in Crowded Scenes." *IEEE CVPR*, 2010. [PDF](http://www.svcl.ucsd.edu/publications/conference/2010/cvpr2010/cvpr_anomaly_2010.pdf)
2. W. Li, V. Mahadevan, N. Vasconcelos. "Anomaly Detection and Localization in Crowded Scenes." *IEEE TPAMI*, 36(1):18-32, 2014. [PDF](http://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf)
3. Dataset: [UCSD Anomaly Dataset on Kaggle](https://www.kaggle.com/datasets/orvile/ucsd-anomaly-dataset/data) (original data from the UCSD Statistical Visual Computing Lab).

## Contributing

Fork, create a feature branch, make a focused change with tests (`python -m pytest`), and open a pull request describing the change and how you validated it.

## License

No license was specified in the supplied materials. Check the repository for a `LICENSE` file before reuse, and consider adding one (e.g. MIT or Apache-2.0). The UCSD dataset has its own terms of use.
