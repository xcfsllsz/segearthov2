# SegEarth-OV-2 Project Memory

## Project Overview

SegEarth-OV is an annotation-free open-vocabulary semantic segmentation framework for remote sensing images (ICCV 2025 / arXiv:2508.18067). It uses CLIP ViT-B/16 + SimFeatUp feature upsampler + Global Bias Alleviation to perform zero-shot segmentation without any pixel-level training annotations.

**Repo**: https://github.com/xcfsllsz/segearthov2 (forked from earth-insights/SegEarth-OV-2)
**Conda env**: `/mnt/data/conda_envs/SegEarth` (Python 3.9, mmseg 1.2.2, torch 2.1.2+cu121)
**GPU**: NVIDIA RTX 3090

---

## Completed Work Summary

### 1. LoveDA Baseline Reproduction

- **Dataset**: `loveda/Val/{Urban,Rural}/images_png/` + `masks_png/` → merged to flat `loveda/val_img/` + `loveda/val_ann/` (1669 pairs)
- **Config**: `configs/cfg_loveda.py` (modified data_prefix paths)
- **Result**: mIoU = 36.9% (matches paper Table 2)
- **Inference command**:
  ```bash
  python eval.py --config ./configs/cfg_loveda.py --work-dir ./work_logs/loveda
  ```
- **Visualization**: `vis_loveda.py` — 3-panel composite (original | GT | prediction), 20 random samples, shared palette

### 2. UAVScenes Dataset Integration (ICCV 2025)

- **Dataset**: UAV drone images, 20 scenes, 24,126 images total, 2048×2448 JPEG
- **Labels**: 26 raw classes (0-25), single-channel PNG. 7 unnamed classes never appear. After cleanup → **11 effective classes**
- **Color map**: Defined in `UAVScenes/cmap.py`, adapted into dataset palettes
- **Data paths**: Symlinked flat structure at `uavscenes_val/images/` + `uavscenes_val/labels/`
- **Eval subsets**: `uavscenes_val_1k/` (1000 images, seed=42), `uavscenes_val_300/` (300 images, seed=42)

**Key custom code**:
- `custom_datasets.py`: `RemapLabels` transform, `UAVScenesDataset` (19c), `UAVScenes13cDataset`, `UAVScenes11cDataset`
- `configs/cls_uavscenes*.txt`: Class prompt files for different class counts
- `configs/cfg_uavscenes*.py`: Eval configs
- `vis_uavscenes.py`: 3-panel visualization for UAVScenes

### 3. Prompt Engineering (prompt_changed.md)

- Designed 3 prompt versions (V0=baseline, V1=detailed, V2=alternative) for all 19 classes
- Reference: `UAVScenes/prompt_changed.md`
- Experiment script: `prompt_search.py` — runs V0/V1/V2 evals, picks per-class best
- **Best prompts per class** saved at `configs/best_cls_uavscenes.txt`
- **Key finding**: V1 style (concrete descriptions like "lawn,crop field,vegetation,pasture" vs simple "green field") won 9/19 classes. V2 won 3/19 (airstrip, river, traffic barrier). V0 kept 7/19.

### 4. Threshold Optimization

- `prob_thd` controls low-confidence pixel suppression to background (bg_idx=0)
- **7-value sweep**: 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
- Experiment script: `threshold_search.py`
- **Optimal**: `prob_thd=0.10` (much lower than LoveDA's 0.3 — UAVScenes has more fine-grained classes)
- At thd=0.10, low-confidence noise filtered into background without destroying rare classes

### 5. Class Pruning Strategy

Removed poorly-performing classes (IoU < 3%) iteratively:

| Deleted classes | Reason |
|---|---|
| dirt road (2), pool (5), bridge (6), traffic barrier (9), umbrella (13), truck (18) | IoU < 3% |
| sidewalk (16), glass roof (17) | IoU < 7% |

### 6. Final Best Configuration

**mIoU = 32.24%** on 1000-image UAVScenes subset (up from 16.14% baseline, **2× improvement**)

| Class | IoU | Class | IoU |
|---|---|---|---|
| river | 72.20 | green field | 67.14 |
| wild field | 45.33 | roof | 43.77 |
| solar panel | 39.02 | car | 20.96 |
| parking lot | 18.11 | paved road | 12.78 |
| container | 12.63 | airstrip | 12.23 |
| background | 10.42 | | |

**Best config files**:
- `configs/cfg_uavscenes_best.py` — 11 classes, prob_thd=0.10
- `configs/cls_uavscenes_best.txt` — 11-class optimal prompts

**Run command**:
```bash
cd /media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2
conda activate /mnt/data/conda_envs/SegEarth
python eval.py --config ./configs/cfg_uavscenes_best.py --work-dir ./work_logs/uavscenes_best
```

---

## Key Files Map

| File | Purpose |
|---|---|
| `eval.py` | Main evaluation script (mmseg Runner.test()) |
| `segearth_segmentor.py` | SegEarthSegmentation model class |
| `custom_datasets.py` | All custom datasets + RemapLabels transform |
| `configs/base_config.py` | Base config (CLIP ViT-B/16, SegEarth mode, SimFeatUp jbu_one) |
| `configs/cfg_loveda.py` | LoveDA eval config |
| `configs/cfg_uavscenes_best.py` | UAVScenes 11c best config |
| `configs/cls_uavscenes_best.txt` | UAVScenes 11c best prompts |
| `vis_loveda.py` | LoveDA 3-panel visualization |
| `vis_uavscenes.py` | UAVScenes 3-panel visualization |
| `prompt_search.py` | Prompt version comparison orchestrator |
| `threshold_search.py` | prob_thd sweep orchestrator |
| `simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt` | SimFeatUp pretrained weights |
| `UAVScenes/` | Dataset repo (ICCV 2025) |
| `UAVScenes/prompt_changed.md` | Prompt candidate reference |
| `UAVScenes/cmap.py` | Original color-ID mapping (26 classes) |

## Evolution Timeline

```
Segearth-OV baseline (LoveDA optical, 7c)     mIoU 36.9%
    ↓
UAVScenes raw baseline (19c, V0, thd=0)       mIoU 16.14%
    ↓  prompt engineering (per-class 3-version search)
+ best prompts (19c, V0+V1+V2 mixed)          mIoU 18.43%
    ↓  threshold sweep [0, 0.3]
+ best thd=0.10 (19c)                        mIoU 18.85%
    ↓  remove 6 dead classes
13 classes                                    mIoU 27.67%
    ↓  remove 2 more weak classes
11 classes (FINAL)                           mIoU 32.24%  ← 2× vs baseline
```

## Important Notes

- `eval.py` sets `HF_ENDPOINT=https://hf-mirror.com` — Chinese mirror for HuggingFace
- CLIP weights auto-download from OpenAI on first run (~400MB ViT-B/16)
- SimFeatUp uses `AdaptiveConv` CUDA extension from featup package
- `.gitignore` blocks `*vis*` patterns — use `git add -f` for visualization scripts
- LoveDA uses `reduce_zero_label=True` (label 0→ignore). UAVScenes uses custom RemapLabels transform instead.
- All evals use sliding window: crop=224, stride=112, resize to 448 long side
- `prob_thd`: threshold for forcing low-confidence pixels to background. =0.0 disables. Optimal for UAVScenes 11c = 0.10.
