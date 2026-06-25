"""Scan prob_thd ∈ [0.0, 0.3] on 300 images with best prompts."""
import subprocess, re, os

THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
PYTHON = "/mnt/data/conda_envs/SegEarth/bin/python"
CLASS_NAMES = [
    "background", "roof", "dirt road", "paved road", "river",
    "pool", "bridge", "container", "airstrip", "traffic barrier",
    "green field", "wild field", "solar panel", "umbrella",
    "glass roof", "parking lot", "sidewalk", "car", "truck",
]
IMG_DIR  = "/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_300/images"
LBL_DIR  = "/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_300/labels"
CLS_PATH = "./configs/best_cls_uavscenes.txt"


def run(thd, label):
    cfg_path = f"./configs/_tmp_thd_{label}.py"
    with open(cfg_path, "w") as f:
        f.write(f"""_base_ = './cfg_uavscenes.py'
model = dict(name_path='{CLS_PATH}', prob_thd={thd})
test_dataloader = dict(dataset=dict(
    data_prefix=dict(img_path='{IMG_DIR}', seg_map_path='{LBL_DIR}'),
    pipeline=[dict(type='LoadImageFromFile'), dict(type='Resize', scale=(448,448), keep_ratio=True),
              dict(type='LoadAnnotations'),
              dict(type='RemapLabels', mapping=[0,1,2,3,4,5,6,255,255,7,8,9,255,10,11,12,13,14,15,16,17,255,255,255,18,255]),
              dict(type='PackSegInputs')]))
""")
    cmd = [PYTHON, "eval.py", "--config", cfg_path, "--work-dir", f"./work_logs/thd/{label}"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    stdout = r.stdout + r.stderr

    per_class, miou, aacc = {}, None, None
    in_table = False
    for line in stdout.split("\n"):
        if "per class results:" in line:
            in_table = True; continue
        if in_table:
            m = re.match(r"\|\s+(.+?)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|", line)
            if m: per_class[m.group(1).strip()] = float(m.group(2))
            elif "aAcc:" in line:
                p = line.split()
                for j, w in enumerate(p):
                    if w == "aAcc:": aacc = float(p[j+1])
                    if w == "mIoU:": miou = float(p[j+1])
                break
    return miou, aacc, per_class


print(f"{'thd':>6s}  {'mIoU':>7s}  {'aAcc':>7s}", end="")
for c in CLASS_NAMES:
    print(f"  {c[:6]:>6s}", end="")
print()

all_results = []
for thd in THRESHOLDS:
    label = f"{thd:.2f}".replace(".", "_")
    miou, aacc, pc = run(thd, label)
    all_results.append((thd, miou, aacc, pc))
    print(f"{thd:6.2f}  {miou:7.2f}  {aacc:7.2f}", end="")
    for c in CLASS_NAMES:
        print(f"  {pc.get(c,0):6.2f}", end="")
    print()

# Find best
best = max(all_results, key=lambda x: x[1])
print(f"\nBest threshold: prob_thd={best[0]:.2f}  mIoU={best[1]:.2f}")
