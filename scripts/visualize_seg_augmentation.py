#!/usr/bin/env python3
"""Generate segmentation augmentation figure for the paper."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEM = "FloorPlan-19-_jpg.rf.0451f0b77e5afd03804179c91070bb88"
IMG_DIR = PROJECT_ROOT / "data" / "segmentation_augmented" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "segmentation_augmented" / "masks"
OUT_DIR = PROJECT_ROOT / "figures"
AUG_SUFFIXES = ["", "_aug01", "_aug03", "_aug06"]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = (128, 128, 128)  # wall
    rgb[mask == 2] = (0, 200, 0)      # room
    return rgb


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colored = mask_to_rgb(mask)
    return cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)


def load_pair(suffix: str):
    img_path = IMG_DIR / f"{STEM}{suffix}.jpg"
    mask_path = MASK_DIR / f"{STEM}{suffix}.png"
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Missing pair for suffix {suffix}")
    return image, mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    titles = [
        "(a) Original",
        "(b) Aug. 01 (geom.)",
        "(c) Aug. 03 (photom.)",
        "(d) Aug. 06 (mixed)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col, (suffix, title) in enumerate(zip(AUG_SUFFIXES, titles)):
        image, mask = load_pair(suffix)
        axes[0, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(title, fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay(image, mask))
        axes[1, col].set_title("Mask overlay", fontsize=10)
        axes[1, col].axis("off")

    fig.suptitle(
        "Segmentation augmentation (FloorPlan-19): image row and label-aligned mask overlay",
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "seg_augmentation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
