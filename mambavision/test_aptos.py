"""
Test / Inference script for MambaVision on the APTOS 2019 test set.
Generates a submission.csv file using a trained checkpoint.

This script supports loading MambaVision models defined in this repository.
It automatically handles finding image directories and checkpoints on Kaggle.

Usage on Kaggle:
    python mambavision/test_aptos.py \
        --checkpoint "/kaggle/working/output_aptos/best.pth" \
        --data-dir "/kaggle/input/datasets/mariaherrerot/aptos2019" \
        --amp
"""
import argparse
import os
import sys
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Make sure we can import the package whether run from repo root or from inside
# the mambavision/ directory.
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

# Auto-mock mamba_ssm if not installed (useful for quick inference on Kaggle/CPU/Windows)
try:
    import mamba_ssm
except ImportError:
    print("mamba_ssm not found. Setting up pure PyTorch mock for inference...")
    import types
    
    # Mock selective_scan_cuda
    mock_cuda = types.ModuleType("selective_scan_cuda")
    sys.modules["selective_scan_cuda"] = mock_cuda

    # Mock torch.utils.serialization
    mock_serial = types.ModuleType("torch.utils.serialization")
    mock_serial_cfg = types.ModuleType("torch.utils.serialization.config")
    class Dummy: pass
    mock_serial_cfg.load = Dummy
    mock_serial_cfg.save = Dummy
    mock_serial.config = mock_serial_cfg
    sys.modules["torch.utils.serialization"] = mock_serial
    sys.modules["torch.utils.serialization.config"] = mock_serial_cfg

    # Mock mamba_ssm modules
    mock_ssm = types.ModuleType("mamba_ssm")
    mock_ssm_ops = types.ModuleType("mamba_ssm.ops")
    mock_ssm_ops_ss = types.ModuleType("mamba_ssm.ops.selective_scan_interface")

    def selective_scan_fn_pytorch(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        import torch.nn.functional as F
        from einops import rearrange, repeat
        
        dtype_in = u.dtype
        u, delta = u.float(), delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
            
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B, C = B.float(), C.float()
            
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
                
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
            
        last_state = None
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, i] if C.dim() == 3 else C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
            
        y = torch.stack(ys, dim=2)
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)

    mock_ssm_ops_ss.selective_scan_fn = selective_scan_fn_pytorch
    sys.modules["mamba_ssm"] = mock_ssm
    sys.modules["mamba_ssm.ops"] = mock_ssm_ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = mock_ssm_ops_ss

from mambavision import create_model  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# Dataset Loader for Testing
# --------------------------------------------------------------------------- #
class TestImageDataset(Dataset):
    """Loads images from a list of samples for inference."""

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


def _find_file(root, filename):
    """Scan the root directory to find the specified file case-insensitively."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == filename.lower():
                return os.path.join(dirpath, fn)
    return None


def _find_image_dir(root, split_keywords):
    """Find a directory of images whose name suggests the given split."""
    candidates = []
    for dirpath, dirnames, _ in os.walk(root):
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


def resolve_test_paths(args):
    """Resolve the test CSV and image directory paths with smart fallbacks."""
    test_csv = args.test_csv
    test_img_dir = args.test_img_dir

    # 1) If paths are not specified, try to find them under args.data_dir or standard inputs
    if not test_csv or not test_img_dir:
        data_root = args.data_dir
        if data_root and os.path.exists(data_root):
            if not test_csv:
                test_csv = _find_file(data_root, "test.csv")
            if not test_img_dir:
                test_img_dir = _find_image_dir(data_root, ["test"]) or _find_image_dir(data_root, ["image", "img"])
        else:
            # Fallback direct checks if data-dir is not specified
            for p in ["/kaggle/input/datasets/mariaherrerot/aptos2019", "/kaggle/input/aptos2019-blindness-detection"]:
                if os.path.exists(p):
                    if not test_csv:
                        test_csv = _find_file(p, "test.csv")
                    if not test_img_dir:
                        test_img_dir = _find_image_dir(p, ["test"]) or _find_image_dir(p, ["image", "img"])
                    break

    # 2) Smart check for missing image directory (scan /kaggle/input)
    if test_img_dir and not os.path.exists(test_img_dir):
        print(f"Warning: Directory '{test_img_dir}' not found. Scanning /kaggle/input for 'test_images' or 'test'...")
        found_dir = _find_image_dir("/kaggle/input", ["test_images"]) or _find_image_dir("/kaggle/input", ["test"])
        if found_dir:
            print(f"  -> Found fallback image directory: {found_dir}")
            test_img_dir = found_dir

    # 3) Resolve nested folder structure (e.g. test_images/test_images)
    if test_img_dir and os.path.exists(test_img_dir):
        nested_cand = os.path.join(test_img_dir, os.path.basename(test_img_dir))
        if os.path.isdir(nested_cand):
            print(f"Detected nested directory. Changing to: {nested_cand}")
            test_img_dir = nested_cand

    return test_csv, test_img_dir


def main():
    parser = argparse.ArgumentParser(description="Inference on APTOS 2019 Test Set using MambaVision")
    # Paths
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth model checkpoint")
    parser.add_argument("--data-dir", default="", help="Root of downloaded dataset")
    parser.add_argument("--test-csv", default="", help="Explicit path to test.csv")
    parser.add_argument("--test-img-dir", default="", help="Explicit path to test_images folder")
    parser.add_argument("--output", default="/kaggle/working/submission.csv", help="Where to save predictions")
    # Model configuration
    parser.add_argument("--model", default="mamba_vision_T", help="Model name if not specified in checkpoint args")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for inference")
    # Execution
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve dataset paths
    test_csv, test_img_dir = resolve_test_paths(args)
    print(f"Resolved test CSV: {test_csv}")
    print(f"Resolved test image directory: {test_img_dir}")

    if not test_img_dir or not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")

    # Load checkpoint info
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Scanning /kaggle/input...")
        found_ckpt = None
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith(".pth") and ("best" in f.lower() or "mamba" in f.lower() or "vision" in f.lower()):
                    found_ckpt = os.path.join(root, f)
                    break
            if found_ckpt:
                break
        if found_ckpt:
            print(f"  -> Found fallback checkpoint at: {found_ckpt}")
            checkpoint_path = found_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint} and no fallback best*.pth found in /kaggle/input")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args) if hasattr(ckpt_args, "__dict__") else {}

    # Extract metadata from checkpoint if available, else fall back to arguments
    num_classes = ckpt_args.get("num_classes", 5)
    img_size = int(ckpt_args.get("img_size", args.img_size))
    model_name = ckpt_args.get("model", args.model)

    print("\nModel configuration:")
    print(f"  - Model name: {model_name}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Image size: {img_size}")

    # Collect test samples
    samples = []
    has_labels = False

    if test_csv and os.path.exists(test_csv):
        print(f"Reading test labels and file IDs from CSV: {test_csv}")
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            lower_header = [h.strip().lower() for h in header]

            # Detect ID and label columns
            id_idx = next((i for i, h in enumerate(lower_header) if any(k in h for k in ["id_code", "id", "image", "name", "file"])), 0)
            label_idx = next((i for i, h in enumerate(lower_header) if any(k in h for k in ["diagnos", "label", "class", "level", "target"])), -1)
            
            has_labels = (label_idx != -1)

            for row in reader:
                if not row or len(row) <= max(id_idx, label_idx):
                    continue
                img_id = row[id_idx].strip()
                label = int(float(row[label_idx].strip())) if has_labels else -1
                
                # Resolve file path
                resolved_path = None
                if os.path.splitext(img_id)[1].lower() in IMG_EXTS:
                    cand = os.path.join(test_img_dir, img_id)
                    if os.path.isfile(cand):
                        resolved_path = cand
                else:
                    for ext in IMG_EXTS:
                        cand = os.path.join(test_img_dir, img_id + ext)
                        if os.path.isfile(cand):
                            resolved_path = cand
                            break
                if resolved_path:
                    samples.append((img_id, resolved_path, label))
    else:
        print(f"No CSV found or provided. Scanning all images directly in {test_img_dir}...")
        for fn in os.listdir(test_img_dir):
            if fn.lower().endswith(IMG_EXTS):
                img_id = os.path.splitext(fn)[0]
                samples.append((img_id, os.path.join(test_img_dir, fn), -1))

    if not samples:
        raise RuntimeError(f"No test images could be loaded from {test_img_dir}")
    print(f"Loaded {len(samples)} test samples.")

    # Setup pipeline
    transform = transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    dataset = TestImageDataset(samples, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers if hasattr(args, "workers") else 4 if os.name != 'nt' else 0,
        pin_memory=True
    )

    # Initialize model
    print(f"Creating MambaVision model: {model_name}")
    model = create_model(model_name, pretrained=False, num_classes=num_classes)

    # Load weights
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    # Run predictions
    predictions = []
    correct = 0
    total = 0

    print("Running inference...")
    with torch.no_grad():
        for ids, images, targets in tqdm(loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            for img_id, pred in zip(ids, preds):
                predictions.append((img_id, int(pred)))

            if has_labels:
                targets_np = targets.numpy()
                valid_mask = targets_np >= 0
                correct += (preds[valid_mask] == targets_np[valid_mask]).sum()
                total += valid_mask.sum()

    # Print accuracy if available
    if has_labels and total > 0:
        print(f"\n=========================================")
        print(f"Độ chính xác (Accuracy) trên tập test: {correct / total * 100:.2f}% ({correct}/{total})")
        print(f"=========================================\n")
    else:
        print("\nLưu ý: Không tìm thấy nhãn thực tế để đo Accuracy. Chỉ lưu kết quả dự đoán.\n")

    # Save output to submission.csv
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "diagnosis"])
        for img_id, pred in predictions:
            writer.writerow([img_id, pred])

    print(f"Hoàn tất! Kết quả dự đoán đã được lưu tại: {args.output}")


if __name__ == "__main__":
    main()
