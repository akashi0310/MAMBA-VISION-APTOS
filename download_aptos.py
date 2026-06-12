"""
Download the APTOS 2019 dataset from Kaggle and inspect its layout.

Usage:
    python download_aptos.py

This script ONLY downloads the data and prints where it landed plus a short
summary of the directory tree / CSV columns, so you can point the training
script (mambavision/train_aptos.py) at the right paths.

Requires:  pip install kagglehub
(Kaggle credentials: kagglehub will prompt / use ~/.kaggle/kaggle.json.)
"""
import os

import kagglehub


def summarize(root, max_entries=20):
    """Print a shallow view of the downloaded dataset directory."""
    print("\nTop-level contents:")
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        kind = "dir " if os.path.isdir(full) else "file"
        print(f"  [{kind}] {name}")

    # Peek into any CSVs to reveal the label columns.
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                csv_path = os.path.join(dirpath, fn)
                rel = os.path.relpath(csv_path, root)
                print(f"\nCSV: {rel}")
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    header = f.readline().strip()
                    print(f"  header: {header}")
                    sample = [f.readline().strip() for _ in range(3)]
                    for s in sample:
                        if s:
                            print(f"  row   : {s}")


def main():
    path = kagglehub.dataset_download("mariaherrerot/aptos2019")
    print("Path to dataset files:", path)
    summarize(path)
    print(
        "\nNext step: pass the path above to the trainer, e.g.\n"
        f'  python mambavision/train_aptos.py --data-dir "{path}" --pretrained'
    )


if __name__ == "__main__":
    main()
