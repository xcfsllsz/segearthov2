"""
UAVScenes three-panel visualization: original | GT | prediction
Same pattern as vis_loveda.py — shared palette, sliding-window inference.
"""
import os, random, traceback
import cv2, mmcv, numpy as np, torch
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
import segearth_segmentor, custom_datasets  # noqa: F401

# ── Label remap (cmap.py 0-25 → 0-18 effective, 255=ignore) ──
LABEL_MAP = np.array([
    0, 1, 2, 3, 4, 5, 6, 255, 255, 7,
    8, 9, 255, 10, 11, 12, 13, 14, 15, 16,
    17, 255, 255, 255, 18, 255,
], dtype=np.uint8)

# ── Palette: index 0=ignore, 1-19=19 classes ──
PALETTE = np.array([
    [0, 0, 0],           # 0: ignore
    [0, 0, 0],           # 1: background
    [119, 11, 32],       # 2: roof
    [180, 165, 180],     # 3: dirt road
    [128, 64, 128],      # 4: paved road
    [173, 216, 230],     # 5: river
    [0, 80, 100],        # 6: pool
    [150, 100, 100],     # 7: bridge
    [250, 170, 30],      # 8: container
    [81, 0, 81],         # 9: airstrip
    [102, 102, 156],     #10: traffic barrier
    [107, 142, 35],      #11: green field
    [210, 180, 140],     #12: wild field
    [220, 220, 0],       #13: solar panel
    [153, 153, 153],     #14: umbrella
    [0, 0, 90],          #15: glass roof
    [250, 170, 160],     #16: parking lot
    [244, 35, 232],      #17: sidewalk
    [0, 0, 142],         #18: car
    [0, 0, 70],          #19: truck
], dtype=np.uint8)

CLASS_NAMES = [
    "ignore", "background", "roof", "dirt road", "paved road",
    "river", "pool", "bridge", "container", "airstrip",
    "traffic barrier", "green field", "wild field", "solar panel",
    "umbrella", "glass roof", "parking lot", "sidewalk", "car", "truck",
]


def main():
    # 1. Model
    cfg = Config.fromfile("./configs/cfg_uavscenes.py")
    print(f"[1/4] Building model: {cfg.model.clip_type}/{cfg.model.vit_type}/{cfg.model.model_type}")
    model = MODELS.build(cfg.model).eval().cuda()
    print("       OK")

    # 2. Sample 20 images
    img_dir = "./uavscenes_val/images"
    ann_dir = "./uavscenes_val/labels"
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.seed(42)
    sampled = random.sample(all_imgs, 20)
    print(f"[2/4] Sampled {len(sampled)} (total: {len(all_imgs)})")

    # 3. Infer + composite
    composites, failed = [], 0
    for idx, name in enumerate(sampled):
        img_path = os.path.join(img_dir, name)
        ann_path = os.path.join(ann_dir, name.replace('.jpg', '.png'))
        try:
            img_rgb = mmcv.imread(img_path, channel_order="rgb")
            img_bgr = mmcv.imread(img_path)
            h, w = img_bgr.shape[:2]

            scale = 448.0 / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img_bgr_rs = mmcv.imresize(img_bgr, (new_w, new_h))

            tensor = torch.from_numpy(img_bgr_rs).permute(2, 0, 1).unsqueeze(0).float()
            ds = SegDataSample()
            ds.set_metainfo({"ori_shape": (h, w), "img_shape": (new_h, new_w),
                             "pad_shape": (new_h, new_w), "img_path": img_path})
            batch = {"inputs": tensor, "data_samples": [ds]}
            pre = model.data_preprocessor(batch, training=False)

            with torch.no_grad():
                pred = model.predict(pre["inputs"].cuda(), pre["data_samples"])
            pred_mask = pred[0].pred_sem_seg.data.cpu().numpy().squeeze().astype(np.uint8)
            pred_color = PALETTE[pred_mask + 1]

            gt_raw = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
            gt_remap = LABEL_MAP[gt_raw]
            gt_color = np.zeros_like(img_rgb)
            for cid in range(19):
                gt_color[gt_remap == cid] = PALETTE[cid + 1]
            gt_color[gt_remap == 255] = PALETTE[0]  # ignore → black

            composite = np.hstack([img_rgb, gt_color, pred_color])
            composites.append((name, composite))
            print(f"       [{idx+1:2d}/20] {name} ({w}×{h}) — OK")
        except Exception:
            failed += 1
            print(f"       [{idx+1:2d}/20] {name} — FAILED\n{traceback.format_exc()}")

    print(f"[3/4] {len(composites)} ok, {failed} failed")
    if not composites:
        return

    # 4. Grid
    n_cols, n_rows = 4, (len(composites) + 3) // 4
    dpi = 100
    fh, fw = composites[0][1].shape[:2]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * fw / dpi * 1.05,
                                                       n_rows * fh / dpi * 1.25))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i, (nm, img) in enumerate(composites):
        axes[i].imshow(img)
        axes[i].set_title(nm, fontsize=5)
        axes[i].axis("off")
    for j in range(len(composites), len(axes)):
        axes[j].axis("off")

    patches = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[c] / 255., ec="black",
                             linewidth=0.2, label=CLASS_NAMES[c]) for c in range(20)]
    fig.legend(handles=patches, loc="lower center", ncol=10, fontsize=5, frameon=False)
    plt.suptitle("UAVScenes — left: original | middle: GT | right: SegEarth-OV",
                 fontsize=10, y=0.99)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.savefig("./uavscenes_vis_20.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[4/4] Saved → ./uavscenes_vis_20.png")


if __name__ == "__main__":
    main()
