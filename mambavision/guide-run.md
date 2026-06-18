# Training MambaVision on APTOS 2019 — Run Guide

Fine-tune a MambaVision backbone on the Kaggle APTOS 2019 classification dataset
(`mariaherrerot/aptos2019`). Two scripts were added:

- [`download_aptos.py`](download_aptos.py) — downloads the dataset and prints its layout.
- [`mambavision/train_aptos.py`](mambavision/train_aptos.py) — single-GPU fine-tuning script.

---

## 1. Install dependencies

From the repo root:

```powershell
pip install -r requirements.txt
pip install kagglehub torchvision
```

> `torch`, `mamba-ssm`, `timm`, `einops`, `Pillow`, etc. come from `requirements.txt`.
> `kagglehub` (for the download) and `torchvision` (for transforms / ImageFolder)
> are extra.

### Kaggle credentials

`kagglehub` needs your Kaggle API token. Get it from
<https://www.kaggle.com/settings> → **Create New Token** (downloads `kaggle.json`),
then place it at:

```
C:\Users\<you>\.kaggle\kaggle.json
```

---

## 2. Download the dataset

```powershell
python download_aptos.py
```

This prints something like:

```
Path to dataset files: C:\Users\<you>\.cache\kagglehub\datasets\mariaherrerot\aptos2019\versions\1
Top-level contents:
  [dir ] train_images
  ...
CSV: train.csv
  header: id_code,diagnosis
  row   : 000c1434d8d7,2
```

**Note two things from this output:**

1. The **path** to the dataset (you'll pass it to the trainer).
2. The **CSV header and label values** — confirm how many classes there are.
   APTOS is often distributed with **5** severity grades (`diagnosis` 0–4). If
   your copy has 5, train with `--num-classes 5` instead of the default `4`.

---

## 3. Train

Basic run (pretrained ImageNet backbone, mixed precision):

```powershell
python mambavision/train_aptos.py --data-dir "<path from step 2>" --pretrained --amp
```

The data loader auto-detects the layout (CSV + image folder, or ImageFolder-style
class subdirectories) and auto-resolves image extensions. If no validation split
exists, it carves 10% off training.

### Common options

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `mamba_vision_T` | Backbone: `mamba_vision_T/T2/S/B/L/L2`. |
| `--pretrained` | off | Load ImageNet weights; the 1000-class head is dropped and a fresh head is trained. |
| `--num-classes` | `4` | Set to `5` if the dataset has 5 grades (check the CSV). |
| `--img-size` | `224` | Input resolution. |
| `--epochs` | `30` | |
| `--batch-size` | `32` | Lower it if you hit CUDA out-of-memory. |
| `--lr` | `5e-4` | |
| `--weight-decay` | `0.05` | |
| `--drop-path` | `0.2` | Stochastic depth. |
| `--amp` | off | Mixed precision (recommended on GPU). |
| `--workers` | `4` | DataLoader workers. |
| `--output` | `./output_aptos` | Where `best.pth` is saved. |

### Example: bigger model, 5 classes, 50 epochs

```powershell
python mambavision/train_aptos.py `
  --data-dir "<path>" `
  --model mamba_vision_S `
  --pretrained --amp `
  --num-classes 5 `
  --epochs 50 --batch-size 16 --lr 3e-4
```

> PowerShell uses a backtick (`` ` ``) for line continuation, as shown above.

---

## Pretrained weights — where they live & loading from any path

When you pass `--pretrained`, the model factory looks for a checkpoint file:

- **If the file exists**, it is loaded directly.
- **If it's missing**, the official ImageNet weights are downloaded to that path
  from NVIDIA's servers.

By default the path is `/tmp/<model>.pth.tar` (e.g. `mamba_vision_T.pth.tar`).
On Windows that resolves to `C:\tmp\<model>.pth.tar`. Note `C:\tmp` is **not
created automatically** — if it doesn't exist the download fails, so create it
first:

```powershell
New-Item -ItemType Directory -Force C:\tmp
```

### Use a checkpoint from any path

Use `--model-path` to load weights from (or download them to) a location you
choose. **`--model-path` only takes effect together with `--pretrained`.**

Load a checkpoint you already downloaded:

```powershell
python mambavision/train_aptos.py `
  --data-dir "<path>" --amp `
  --pretrained --model-path "D:\weights\mamba_vision_T.pth.tar"
```

Download the official weights to a folder of your choice (used as a cache on the
next run, so it only downloads once):

```powershell
New-Item -ItemType Directory -Force .\weights
python mambavision/train_aptos.py `
  --data-dir "<path>" --amp `
  --model mamba_vision_T `
  --pretrained --model-path ".\weights\mamba_vision_T.pth.tar"
```

> Make sure `--model-path` matches `--model`: a `mamba_vision_S` checkpoint will
> not load into a `mamba_vision_T` model. Official weights for each variant can
> also be downloaded by hand from the
> [MambaVision Hugging Face collection](https://huggingface.co/collections/nvidia/mambavision-66943871a6b36c9e78b327d3).

The 1000-class ImageNet head in the checkpoint is dropped automatically (it does
not match your `--num-classes` head); the rest of the backbone is loaded.

---

## 4. Output

Per-epoch metrics are printed:

```
Epoch   1/30 | train loss 1.2031 acc 0.5512 | val loss 0.9842 acc 0.6701 | lr 5.00e-04 | 73.2s
  -> saved new best (0.6701) to ./output_aptos\best.pth
```

The best validation checkpoint is saved to `./output_aptos/best.pth`, containing
`state_dict`, `epoch`, `acc`, and the run `args`.

---

## Overriding auto-detection

If the layout isn't detected correctly, pass paths explicitly.

CSV layout:

```powershell
python mambavision/train_aptos.py `
  --train-csv "<...>\train.csv" --train-img-dir "<...>\train_images" `
  --val-csv   "<...>\valid.csv" --val-img-dir   "<...>\val_images" `
  --pretrained --amp
```

ImageFolder layout (`<dir>/<class>/*.png`):

```powershell
python mambavision/train_aptos.py `
  --train-dir "<...>\train" --val-dir "<...>\val" `
  --pretrained --amp
```

---

## Troubleshooting

- **`No module named 'timm'` / `mamba_ssm`** — run `pip install -r requirements.txt`.
- **`mamba-ssm` build fails on Windows** — it needs a CUDA toolchain; consider WSL2
  or a Linux machine if installation fails.
- **CUDA out of memory** — reduce `--batch-size` (e.g. `16` or `8`) and/or `--img-size`.
- **"Could not auto-detect dataset layout"** — inspect the printed tree from
  `download_aptos.py` and use the explicit `--train-csv`/`--train-img-dir` (or
  `--train-dir`/`--val-dir`) flags above.
- **Label/class-count warning** — the dataset has more classes than `--num-classes`;
  raise `--num-classes` to match.
