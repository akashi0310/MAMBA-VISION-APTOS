"""
Binary Evaluation Script for APTOS 2019 using MambaVision or FastViT.
Merges classes [1, 2, 3, 4] into Class 1 (DR Present) vs Class 0 (No DR)
to evaluate binary Recall (Sensitivity), Specificity, Precision, F1, and Confusion Matrix.

Usage on Kaggle:
    python mambavision/eval_binary_aptos.py \
        --checkpoint "/kaggle/working/output_mamba/best.pth" \
        --data-dir "/kaggle/input/datasets/mariaherrerot/aptos2019" \
        --amp
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add package directory to path
model_found = False
if '__file__' in globals():
    path_cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(path_cand, "mambavision")):
        sys.path.insert(0, path_cand)
        model_found = True

if not model_found and os.path.exists("mambavision"):
    sys.path.insert(0, os.getcwd())
    model_found = True

if not model_found:
    for root, dirs, files in os.walk(os.getcwd()):
        if "mambavision" in dirs:
            sys.path.insert(0, root)
            model_found = True
            break

from mambavision import create_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class EvalDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img_id, img, label


def find_dataset_files(root):
    train_csv = None
    val_csv = None
    test_csv = None
    img_dir = None

    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".csv"):
                full = os.path.join(dirpath, fn)
                if "val" in low or "valid" in low:
                    val_csv = full
                elif "test" in low:
                    test_csv = full
                elif "train" in low:
                    train_csv = full
        for d in dirnames:
            low = d.lower()
            if any(k in low for k in ["images", "train", "val", "test", "img"]):
                full = os.path.join(dirpath, d)
                if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(full) if os.path.isfile(os.path.join(full, f))):
                    if img_dir is None or "test" in low or "val" in low:
                        img_dir = full

    return train_csv, val_csv, test_csv, img_dir


def load_samples_from_csv(csv_path, img_dir):
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        lower_header = [h.strip().lower() for h in header]

        id_idx = next((i for i, h in enumerate(lower_header) if any(k in h for k in ["id_code", "id", "image", "name", "file"])), 0)
        label_idx = next((i for i, h in enumerate(lower_header) if any(k in h for k in ["diagnos", "label", "class", "level", "target"])), 1)

        for row in reader:
            if not row or len(row) <= max(id_idx, label_idx):
                continue
            img_id = row[id_idx].strip()
            label = int(float(row[label_idx].strip()))

            resolved_path = None
            if os.path.splitext(img_id)[1].lower() in IMG_EXTS:
                cand = os.path.join(img_dir, img_id)
                if os.path.isfile(cand):
                    resolved_path = cand
            else:
                for ext in IMG_EXTS:
                    cand = os.path.join(img_dir, img_id + ext)
                    if os.path.isfile(cand):
                        resolved_path = cand
                        break
            if resolved_path:
                samples.append((img_id, resolved_path, label))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Binary Evaluation (No DR vs DR Present) for APTOS 2019")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pth checkpoint")
    parser.add_argument("--data-dir", default="", help="Dataset root directory")
    parser.add_argument("--csv-path", default="", help="Explicit path to CSV file (train.csv / val.csv / test.csv)")
    parser.add_argument("--img-dir", default="", help="Explicit path to image folder")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Resolve CSV and image directory
    csv_path = args.csv_path
    img_dir = args.img_dir
    if not csv_path or not img_dir:
        root = args.data_dir or "/kaggle/input/datasets/mariaherrerot/aptos2019"
        tr_csv, va_csv, te_csv, auto_img = find_dataset_files(root)
        csv_path = csv_path or va_csv or te_csv or tr_csv
        img_dir = img_dir or auto_img

    print(f"Evaluating on CSV: {csv_path}")
    print(f"Image Directory: {img_dir}")

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file at {csv_path}")

    samples = load_samples_from_csv(csv_path, img_dir)
    print(f"Loaded {len(samples)} evaluation samples.")

    # 2. Load Checkpoint & Model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        ckpt_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        ckpt_args = {}

    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args) if hasattr(ckpt_args, "__dict__") else {}

    model_name = ckpt_args.get("model", "mamba_vision_S")
    num_classes = ckpt_args.get("num_classes", 5)
    img_size = int(ckpt_args.get("img_size", args.img_size))

    model = create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 3. Setup DataLoader
    transform = transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = EvalDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Inference & Binary Mapping
    y_true_5class = []
    y_pred_5class = []

    print("Running model inference...")
    with torch.no_grad():
        for _, images, targets in tqdm(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred_5class.extend(preds)
            y_true_5class.extend(targets.numpy())

    y_true_5class = np.array(y_true_5class)
    y_pred_5class = np.array(y_pred_5class)

    # --------------------------------------------------------------------------- #
    # Binary Merge Logic: Class 0 = No DR (0), Classes 1,2,3,4 = DR Present (1)
    # --------------------------------------------------------------------------- #
    y_true_binary = (y_true_5class > 0).astype(int)
    y_pred_binary = (y_pred_5class > 0).astype(int)

    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        recall_score,
        precision_score,
        f1_score,
        accuracy_score,
    )

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    acc = accuracy_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)  # Sensitivity: TP / (TP + FN)
    specificity = tn / (tn + fp)  # Specificity: TN / (TN + FP)
    precision = precision_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    print("\n=======================================================")
    print("           BINARY EVALUATION RESULTS                   ")
    print("      Class 0 (No DR) vs Classes 1, 2, 3, 4 (DR)       ")
    print("=======================================================")
    print(f"Total Samples           : {len(y_true_binary)}")
    print(f"No DR (Class 0)         : {np.sum(y_true_binary == 0)}")
    print(f"DR Present (Class 1-4)  : {np.sum(y_true_binary == 1)}")
    print("-------------------------------------------------------")
    print(f"Accuracy                : {acc * 100:.2f}%")
    print(f"Recall (Sensitivity)    : {recall * 100:.2f}%  <-- (TP / (TP + FN))")
    print(f"Specificity             : {specificity * 100:.2f}%  <-- (TN / (TN + FP))")
    print(f"Precision               : {precision * 100:.2f}%")
    print(f"F1-Score                : {f1 * 100:.2f}%")
    print("-------------------------------------------------------")
    print("Binary Confusion Matrix:")
    print(f"  True Negatives (TN - Healthy predicted Healthy) : {tn}")
    print(f"  False Positives (FP - Healthy predicted DR)    : {fp}")
    print(f"  False Negatives (FN - DR predicted Healthy)    : {fn}  <-- MISSED DR CASES")
    print(f"  True Positives (TP - DR predicted DR)          : {tp}")
    print("=======================================================")

    print("\nDetailed Binary Classification Report:")
    print(classification_report(y_true_binary, y_pred_binary, target_names=["No DR (Class 0)", "DR Present (Class 1-4)"]))


if __name__ == "__main__":
    main()
