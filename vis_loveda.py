"""
Visually evaluate SegEarth-OV on LoveDA via randomly sampled 20 images.
Produces a three-panel composite grid:
  left=original  |  middle=GT label  |  right=prediction

GT and prediction share the **exact same** color palette for fair comparison.
Reuses the **exact** model config and inference logic from eval.py.
"""
import os
import random
import traceback

import cv2
import mmcv
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

# ── register SegEarthSegmentation + custom datasets with mmseg ──
import segearth_segmentor  # noqa: F401  (side-effect: @MODELS.register_module)
import custom_datasets     # noqa: F401  (side-effect: @DATASETS.register_module)


# ═══════════════════════════════════════════════════════════════
# LoveDA class palette  (matches mmseg LoveDADataset.METAINFO)
# Index: 0=ignore(black), 1=background, 2=building, 3=road,
#        4=water, 5=barren, 6=forest, 7=agricultural
# ═══════════════════════════════════════════════════════════════
PALETTE = np.array([
    [0,   0,   0  ],   # 0: ignore / no-data
    [255, 255, 255],   # 1: background
    [255, 0,   0  ],   # 2: building
    [255, 255, 0  ],   # 3: road
    [0,   0,   255],   # 4: water
    [159, 129, 183],   # 5: barren
    [0,   255, 0  ],   # 6: forest
    [255, 195, 128],   # 7: agricultural
], dtype=np.uint8)

CLASS_NAMES = [
    "ignore", "background", "building", "road",
    "water", "barren", "forest", "agricultural",
]


def main():
    # ──────────────────────────────────────────────
    # 1. Load config & build model (same as eval.py)
    # ──────────────────────────────────────────────
    cfg = Config.fromfile("./configs/cfg_loveda.py")
    print(f"[1/4] Building model: CLIP={cfg.model.clip_type}, "
          f"ViT={cfg.model.vit_type}, mode={cfg.model.model_type}")
    model = MODELS.build(cfg.model)
    model.eval().cuda()
    print("       Model built successfully.")

    # ──────────────────────────────────────────────
    # 2. Randomly sample 20 validation images
    # ──────────────────────────────────────────────
    img_dir = "./loveda/val_img"
    ann_dir = "./loveda/val_ann"
    all_imgs = sorted(os.listdir(img_dir))
    random.seed(42)
    sampled = random.sample(all_imgs, 20)
    print(f"[2/4] Sampled {len(sampled)} images "
          f"(total {len(all_imgs)} available)")

    # ──────────────────────────────────────────────
    # 3. Inference + three-panel composites
    # ──────────────────────────────────────────────
    composites = []   # list of (filename, rgb_numpy_array)
    failed = 0

    for idx, img_name in enumerate(sampled):
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, img_name)
        try:
            # ---- 3a. Read images ----
            img_rgb = mmcv.imread(img_path, channel_order="rgb")   # for display
            img_bgr = mmcv.imread(img_path)                        # for model
            h, w = img_bgr.shape[:2]

            # ---- 3b. Resize BGR for model input (448 long side) ----
            scale = 448.0 / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img_bgr_resized = mmcv.imresize(img_bgr, (new_w, new_h))

            # ---- 3c. Preprocess & predict ----
            img_tensor = (
                torch.from_numpy(img_bgr_resized)
                .permute(2, 0, 1).unsqueeze(0).float()
            )
            data_sample = SegDataSample()
            data_sample.set_metainfo({
                "ori_shape": (h, w),
                "img_shape": (new_h, new_w),
                "pad_shape": (new_h, new_w),
                "img_path": img_path,
            })
            batch = {"inputs": img_tensor, "data_samples": [data_sample]}
            preprocessed = model.data_preprocessor(batch, training=False)

            with torch.no_grad():
                seg_pred = model.predict(
                    preprocessed["inputs"].cuda(),
                    preprocessed["data_samples"],
                )
            pred_mask = (
                seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8)
            )
            # pred_mask: 0-6 class indices → shift +1 for extended palette
            pred_color = PALETTE[pred_mask + 1]

            # ---- 3d. Read & colorize GT (single-channel label) ----
            # GT raw: 0=ignore, 1=bg, 2=bld, 3=road, 4=water, 5=barren, 6=forest, 7=agri
            # Maps directly to PALETTE indices 0-7
            gt_raw = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            gt_color = PALETTE[gt_raw]

            # ---- 3e. Three-panel composite ----
            composite = np.hstack([img_rgb, gt_color, pred_color])
            composites.append((img_name, composite))

            print(f"       [{idx+1:2d}/20] {img_name}  ({w}×{h}) — OK")

        except Exception:
            failed += 1
            print(f"       [{idx+1:2d}/20] {img_name} — FAILED:\n"
                  f"        {traceback.format_exc()}")

    print(f"[3/4] Inference done. {len(composites)} succeeded, {failed} failed.")

    if not composites:
        print("       No successful composites, aborting.")
        return

    # ──────────────────────────────────────────────
    # 4. Plot grid & save
    # ──────────────────────────────────────────────
    n_cols = 4
    n_rows = (len(composites) + n_cols - 1) // n_cols

    dpi = 100
    first_h, first_w = composites[0][1].shape[:2]
    cell_w = first_w / dpi * 1.05
    cell_h = first_h / dpi * 1.25

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (name, img) in enumerate(composites):
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=6)
        axes[i].axis("off")

    for j in range(len(composites), len(axes)):
        axes[j].axis("off")

    # ── Legend ──
    legend_patches = []
    for ci, (cname, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        legend_patches.append(
            plt.Rectangle((0, 0), 1, 1, fc=color / 255.0,
                          ec="black", linewidth=0.2, label=f"{ci}: {cname}")
        )
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(CLASS_NAMES), fontsize=6.5, frameon=False)

    plt.suptitle("SegEarth-OV on LoveDA  |  "
                 "left: original  |  middle: GT  |  right: prediction",
                 fontsize=10, y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    out_path = "./loveda_vis_20.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[4/4] Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
