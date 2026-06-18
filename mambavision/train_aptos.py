"""
Fine-tune a MambaVision backbone on the APTOS 2019 dataset (classification).

This is a lightweight, single-process training script (no DDP / torchrun needed)
meant to run on a single GPU on Windows. It reuses the model definitions in this
repo through ``create_model`` so the MambaVision backbone is identical to the one
used by the original ``train.py``.

Download the data first with ``download_aptos.py`` (see that file). Then:

    python mambavision/train_aptos.py --data-dir "<path printed by download_aptos.py>" --pretrained

The data loader auto-detects two common layouts:

  1. CSV layout (typical for APTOS re-uploads):
        <data-dir>/train.csv        columns like: id_code, diagnosis
        <data-dir>/train_images/    *.png named by id_code
     (and similar valid/test CSVs + image folders)

  2. ImageFolder layout:
        <data-dir>/train/<class>/*.png
        <data-dir>/val/<class>/*.png

You can also point at things explicitly with --train-csv/--val-csv +
--train-img-dir/--val-img-dir, or --train-dir/--val-dir for ImageFolder.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Make sure we can import the package whether run from repo root or from inside
# the mambavision/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mambavision import create_model  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CsvImageDataset(Dataset):
    """Reads (image, label) pairs from a CSV file.

    Auto-detects the id column and the label column, and resolves the on-disk
    image filename (with or without an explicit extension).
    """

    def __init__(self, csv_path, img_dir, transform=None):
        import csv

        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            id_idx, label_idx = self._detect_columns(header)
            for row in reader:
                if not row or len(row) <= max(id_idx, label_idx):
                    continue
                img_id = row[id_idx].strip()
                label = int(float(row[label_idx].strip()))
                path = self._resolve_path(img_id)
                if path is not None:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(
                f"No usable rows found in {csv_path} (img_dir={img_dir}). "
                "Check that the image directory matches the CSV ids."
            )
        self.num_classes = max(lbl for _, lbl in self.samples) + 1

    @staticmethod
    def _detect_columns(header):
        lower = [h.strip().lower() for h in header]

        def find(keys, default):
            for i, h in enumerate(lower):
                if any(k in h for k in keys):
                    return i
            return default

        id_idx = find(["id_code", "id", "image", "name", "code", "file"], 0)
        label_idx = find(["diagnos", "label", "class", "level", "target"], 1)
        if id_idx == label_idx:
            label_idx = 1 if id_idx == 0 else 0
        return id_idx, label_idx

    def _resolve_path(self, img_id):
        # Already has a valid extension?
        if os.path.splitext(img_id)[1].lower() in IMG_EXTS:
            cand = os.path.join(self.img_dir, img_id)
            return cand if os.path.isfile(cand) else None
        # Otherwise try appending known extensions.
        for ext in IMG_EXTS:
            cand = os.path.join(self.img_dir, img_id + ext)
            if os.path.isfile(cand):
                return cand
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# Layout resolution
# --------------------------------------------------------------------------- #
def _walk_find(root, predicate):
    """Find the first path under root for which predicate(name) is True."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in sorted(dirnames) + sorted(filenames):
            if predicate(name.lower()):
                return os.path.join(dirpath, name)
    return None


def _find_csv(root, split_keywords):
    """Find a CSV whose name suggests the given split (train/val/test)."""
    best = None
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            low = fn.lower()
            if low.endswith(".csv") and any(k in low for k in split_keywords):
                return os.path.join(dirpath, fn)
    return best


def _find_image_dir(root, split_keywords):
    """Find a directory of images whose name suggests the given split."""
    candidates = []
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in dirnames:
            low = d.lower()
            if any(k in low for k in split_keywords):
                full = os.path.join(dirpath, d)
                # Prefer dirs that actually contain images.
                has_img = any(
                    f.lower().endswith(IMG_EXTS)
                    for f in os.listdir(full)
                    if os.path.isfile(os.path.join(full, f))
                )
                candidates.append((has_img, full))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (not c[0]))  # image-containing dirs first
    return candidates[0][1]


def build_datasets(args, train_tf, val_tf):
    """Resolve the dataset layout and return (train_ds, val_ds)."""
    # 1) Explicit ImageFolder dirs.
    if args.train_dir and args.val_dir:
        print(f"Using ImageFolder layout: {args.train_dir} | {args.val_dir}")
        return ImageFolder(args.train_dir, train_tf), ImageFolder(args.val_dir, val_tf)

    # 2) Explicit CSV layout.
    if args.train_csv and args.train_img_dir:
        train_ds = CsvImageDataset(args.train_csv, args.train_img_dir, train_tf)
        val_csv = args.val_csv or args.train_csv
        val_img = args.val_img_dir or args.train_img_dir
        val_ds = CsvImageDataset(val_csv, val_img, val_tf)
        return train_ds, val_ds

    # 3) Auto-detect under --data-dir.
    root = args.data_dir
    if not root:
        raise SystemExit("Provide --data-dir, or explicit --train-* paths.")

    train_kw = ["train"]
    val_kw = ["val", "valid"]
    test_kw = ["test"]

    train_csv = _find_csv(root, train_kw)
    val_csv = _find_csv(root, val_kw) or _find_csv(root, test_kw)

    if train_csv:
        print(f"Auto-detected CSV layout.\n  train csv: {train_csv}")
        train_img = args.train_img_dir or _find_image_dir(root, train_kw) \
            or _find_image_dir(root, ["image", "img"])
        val_img = args.val_img_dir or _find_image_dir(root, val_kw) \
            or _find_image_dir(root, test_kw) or train_img
        if not train_img:
            raise SystemExit(
                "Found a train CSV but no image directory. "
                "Pass --train-img-dir explicitly."
            )
        print(f"  train images: {train_img}")
        train_ds = CsvImageDataset(train_csv, train_img, train_tf)
        if val_csv:
            print(f"  val csv: {val_csv}\n  val images: {val_img}")
            val_ds = CsvImageDataset(val_csv, val_img, val_tf)
        else:
            print("  No val CSV found -> splitting 10% of train for validation.")
            train_ds, val_ds = _random_split(train_ds, args, val_tf)
        return train_ds, val_ds

    # 4) Auto-detect ImageFolder layout (train/ + val/ subdirs).
    train_dir = _find_image_dir(root, train_kw)
    val_dir = _find_image_dir(root, val_kw) or _find_image_dir(root, test_kw)
    if train_dir and val_dir:
        print(f"Auto-detected ImageFolder layout:\n  {train_dir}\n  {val_dir}")
        return ImageFolder(train_dir, train_tf), ImageFolder(val_dir, val_tf)

    raise SystemExit(
        f"Could not auto-detect dataset layout under {root}. "
        "Inspect it with download_aptos.py and pass explicit "
        "--train-csv/--train-img-dir (or --train-dir/--val-dir)."
    )


def _random_split(train_ds, args, val_tf):
    """Carve a validation subset out of a single training dataset."""
    n = len(train_ds)
    n_val = max(1, int(n * 0.1))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    # Wrap so the val subset uses val transforms.
    val_view = CsvImageDataset.__new__(CsvImageDataset)
    val_view.img_dir = train_ds.img_dir
    val_view.transform = val_tf
    val_view.samples = [train_ds.samples[i] for i in val_idx]
    val_view.num_classes = train_ds.num_classes
    train_ds.samples = [train_ds.samples[i] for i in train_idx]
    return train_ds, val_view


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, scaler, device, train, amp):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MambaVision on APTOS 2019")
    # Data
    parser.add_argument("--data-dir", default="", help="Root of downloaded dataset (auto-detect layout)")
    parser.add_argument("--train-csv", default="")
    parser.add_argument("--val-csv", default="")
    parser.add_argument("--train-img-dir", default="")
    parser.add_argument("--val-img-dir", default="")
    parser.add_argument("--train-dir", default="", help="ImageFolder train dir")
    parser.add_argument("--val-dir", default="", help="ImageFolder val dir")
    # Model
    parser.add_argument("--model", default="mamba_vision_T")
    parser.add_argument("--pretrained", action="store_true", help="Load ImageNet pretrained backbone")
    parser.add_argument(
        "--model-path", default="",
        help="Path to a pretrained .pth.tar checkpoint. If it exists it is loaded; "
             "if missing (and --pretrained is set) the official weights are downloaded here. "
             "Defaults to /tmp/<model>.pth.tar when omitted.",
    )
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--drop-path", type=float, default=0.2)
    # Optimization
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Mixed-precision training")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./output_aptos", help="Where to save checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds, val_ds = build_datasets(args, train_tf, val_tf)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Sanity-check the label range vs --num-classes.
    detected = getattr(train_ds, "num_classes", None)
    if detected is not None and detected > args.num_classes:
        print(
            f"WARNING: dataset has labels up to class {detected - 1} "
            f"({detected} classes) but --num-classes={args.num_classes}. "
            "Adjust --num-classes if needed."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    create_kwargs = dict(
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
    )
    if args.model_path:
        # Threads through to the model factory's `model_path` kwarg: an existing
        # file is loaded, a missing one is downloaded to this path.
        create_kwargs["model_path"] = args.model_path
    model = create_model(args.model, **create_kwargs).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True, amp=args.amp
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False, amp=args.amp
        )
        scheduler.step()
        dt = time.time() - t0
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            ckpt = os.path.join(args.output, "best.pth")
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "acc": best_acc, "args": vars(args)},
                ckpt,
            )
            print(f"  -> saved new best ({best_acc:.4f}) to {ckpt}")

    print(f"Done. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
