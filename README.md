# KGC-UKAN

**A Kolmogorov–Arnold U-Net with KAN-Gated Cross-Scale Context and Polarity-Aware Uncertainty Skips for Medical Image Segmentation**

This repository contains the official implementation of KGC-UKAN.

---

## Architecture

<p align="center">
  <img src="figures/architecture.png" width="95%" alt="KGC-UKAN architecture"/>
</p>

KGC-UKAN is a five-stage encoder–decoder built on Kolmogorov–Arnold layers with three task-specific modules:

- **SAKE** — Spline-Adaptive Kolmogorov Edge block (deep encoder/decoder stages).
- **KG-CSA** — KAN-Gated Cross-Scale ASPP bottleneck.
- **PU-LASk** — Polarity-aware Uncertainty Linear-Attention skip connections.

<p align="center">
  <img src="figures/sake.png" width="32%" alt="SAKE"/>
  <img src="figures/kgcsa.png" width="32%" alt="KG-CSA"/>
  <img src="figures/pulask.png" width="32%" alt="PU-LASk"/>
</p>

> Place the figure files in a `figures/` folder (`architecture.png`, `sake.png`, `kgcsa.png`, `pulask.png`).

---

## Reproducibility

### 1. Environment

```bash
git clone https://github.com/Hassan48khan/KGC-UKAN.git
cd KGC-UKAN
pip install -r requirements.txt
```

### 2. Data layout

Place each dataset under `inputs/<dataset>/` with matching basenames:

```
inputs/
  busi/
    images/   img1.png  img2.png ...
    masks/    img1.png  img2.png ...
```

Supported datasets in the paper: BUSI, CVC-ClinicDB, BRISC, COVID-19 CT, Chest X-ray (TB), ISIC 2018, MCE. All images and masks are resized to the chosen input resolution.

### 3. Training

All experiments use Adam (initial LR `1e-4`, cosine annealing to `1e-5`),
batch size 16, 300 epochs, and an 80/20 train/val split. Augmentation is
random horizontal/vertical flip and rotation. Results are averaged over five
independent runs with different seeds.

512×512 (default, as in the paper):

```bash
python train.py --dataset busi --data_dir inputs \
    --input_w 512 --input_h 512 --aspp_rates 6 12 18 \
    --batch_size 16 --epochs 300 --lr 1e-4 --min_lr 1e-5 \
    --name kgc_ukan_busi
```

256×256 (lighter setting):

```bash
python train.py --dataset busi --data_dir inputs \
    --input_w 256 --input_h 256 --aspp_rates 2 4 6 \
    --name kgc_ukan_busi_256
```

Reproduce the five runs by repeating with `--seed {0,1,2,3,4}`.

### 4. Evaluation

```bash
python val.py --name kgc_ukan_busi --dataset busi --data_dir inputs \
    --input_w 512 --input_h 512 --aspp_rates 6 12 18
```

Reports IoU, Dice, HD95, and F1.

### 5. Cross-dataset generalization

Evaluate a model trained on a source dataset directly on a target dataset (no fine-tuning):

```bash
python val.py --dataset cvc --data_dir inputs \
    --checkpoint models/kgc_ukan_busi/model_best.pth \
    --input_w 512 --input_h 512 --aspp_rates 6 12 18 --eval_all True
```

### 6. Ablations

Toggle individual modules to reproduce the ablation study:

```bash
# without SAKE edge branch
python train.py --dataset busi --use_edge False  --name abl_no_sake
# without PU-LASk skips (plain additive skips)
python train.py --dataset busi --use_pulask False --name abl_no_pulask
# without deep supervision
python train.py --dataset busi --deep_supervision False --name abl_no_ds
```

### 7. Complexity (GFLOPs / Params)

```bash
python - <<'PY'
import torch
from kgc_ukan import KGC_UKAN
from thop import profile, clever_format
m = KGC_UKAN(num_classes=1, img_size=512, embed_dims=[128,160,256],
             aspp_rates=(6,12,18), deep_supervision=False).eval()
flops, params = profile(m, inputs=(torch.randn(1,3,512,512),), verbose=False)
print(clever_format([flops, params], "%.2f"))
PY
```

### Key hyper-parameters

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Initial LR / Min LR | 1e-4 / 1e-5 (cosine annealing) |
| Batch size | 16 |
| Epochs | 300 |
| Input size | 512×512 (or 256×256) |
| `aspp_rates` | (6,12,18) at 512 / (2,4,6) at 256 |
| Train/val split | 80 / 20 |
| Runs | 5 (seeds 0–4) |

---

## License

Released under the [MIT License](LICENSE).
