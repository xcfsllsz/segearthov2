"""
Prompt engineering search for UAVScenes.
Tests 3 prompt versions (V0/V1/V2) on the same 1000-image subset,
picks the best per-class prompt, and validates the final combination.
"""
import subprocess, re, os, sys

VERSIONS = ["v0", "v1", "v2"]
SUBSET_CFG = "./configs/cfg_uavscenes_1k.py"
WORK_DIR  = "./work_logs/prompt_search"
CLASS_NAMES = [
    "background", "roof", "dirt road", "paved road", "river",
    "pool", "bridge", "container", "airstrip", "traffic barrier",
    "green field", "wild field", "solar panel", "umbrella",
    "glass roof", "parking lot", "sidewalk", "car", "truck",
]
PYTHON = "/mnt/data/conda_envs/SegEarth/bin/python"


def run_eval(cls_name, label):
    """Run eval.py with given cls file, return (mIoU, aAcc, per_class_IoU_dict)."""
    cfg = SUBSET_CFG.replace(".py", f"_{label}.py")
    wdir = f"{WORK_DIR}/{label}"
    cmd = [PYTHON, "eval.py", "--config", cfg, "--work-dir", wdir]
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
    stdout = result.stdout + result.stderr
    print(stdout[-2000:])  # tail

    # Parse per-class IoU table
    per_class = {}
    mIoU = aAcc = None
    in_table = False
    for line in stdout.split("\n"):
        if "per class results:" in line:
            in_table = True
            continue
        if in_table:
            m = re.match(r"\|\s+(.+?)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|", line)
            if m:
                cname = m.group(1).strip()
                iou = float(m.group(2))
                acc = float(m.group(3))
                per_class[cname] = iou
            elif "aAcc:" in line:
                parts = line.split()
                for j, p in enumerate(parts):
                    if p == "aAcc:":
                        aAcc = float(parts[j + 1])
                    if p == "mIoU:":
                        mIoU = float(parts[j + 1])
                break
    return mIoU, aAcc, per_class


def write_subset_cfg(cls_path, label):
    """Write a temporary config pointing to the 1k subset + given cls file."""
    content = f"""_base_ = './cfg_uavscenes.py'

model = dict(
    name_path='{cls_path}',
    prob_thd=0.0,
)

test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_1k/images',
            seg_map_path='/media/zero/DAAA8605AA85DE7F1/SegEarth-OV-2/uavscenes_val_1k/labels'),
    )
)
"""
    cfg_path = SUBSET_CFG.replace(".py", f"_{label}.py")
    with open(cfg_path, "w") as f:
        f.write(content)
    return cfg_path


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    results = {}  # {class_name: {v0: iou, v1: iou, v2: iou}}

    # ── Phase 1: run V0, V1, V2 ──
    for ver in VERSIONS:
        cls_path = f"./configs/cls_uavscenes_{ver}.txt"
        write_subset_cfg(cls_path, ver)
        miou, aacc, per_class = run_eval(cls_path, ver)
        print(f"\n  {ver}: mIoU={miou}, aAcc={aacc}")
        for cname, iou in per_class.items():
            results.setdefault(cname, {})[ver] = iou

    # ── Phase 2: pick best per class ──
    best_lines = []
    print(f"\n{'='*60}\nPer-class best prompt selection:\n{'='*60}")
    print(f"{'Class':<20s} {'V0':>7s} {'V1':>7s} {'V2':>7s}  {'Best':>5s}")
    for cname in CLASS_NAMES:
        scores = {ver: results.get(cname, {}).get(ver, 0) for ver in VERSIONS}
        best_ver = max(scores, key=scores.get)
        best_iou = scores[best_ver]
        print(f"{cname:<20s} {scores['v0']:7.2f} {scores['v1']:7.2f} {scores['v2']:7.2f}  → {best_ver}: {best_iou:.2f}")

        # Read the best prompt line for this class
        cls_path = f"./configs/cls_uavscenes_{best_ver}.txt"
        with open(cls_path) as f:
            lines = f.read().strip().split("\n")
        # Find matching line by position (classes in order)
        idx = CLASS_NAMES.index(cname)
        best_lines.append(lines[idx] if idx < len(lines) else CLASS_NAMES[idx])

    # ── Phase 3: generate best cls ──
    best_cls_path = "./configs/best_cls_uavscenes.txt"
    with open(best_cls_path, "w") as f:
        f.write("\n".join(best_lines) + "\n")
    print(f"\nBest cls written to {best_cls_path}")

    # ── Phase 4: final validation ──
    write_subset_cfg(best_cls_path, "best")
    miou, aacc, per_class = run_eval(best_cls_path, "best")
    print(f"\n{'='*60}")
    print(f"FINAL RESULT with best prompts:")
    print(f"  mIoU = {miou}")
    print(f"  aAcc = {aacc}")

    # ── Summary ──
    v0_path = f"{WORK_DIR}/v0/results.txt"
    with open(v0_path) as f:
        v0_str = f.read().strip()
    v0_miou = float(re.search(r"mIoU:\s*([\d.]+)", v0_str).group(1))

    summary_path = f"{WORK_DIR}/summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Baseline mIoU (V0): {v0_miou:.2f}\n")
        f.write(f"Optimized mIoU:     {miou:.2f}\n")
        f.write(f"Delta:              {miou - v0_miou:+.2f}\n")
        f.write(f"\nBest per-class prompts:\n")
        for i, (cname, line) in enumerate(zip(CLASS_NAMES, best_lines)):
            best_ver = max(VERSIONS, key=lambda v: results.get(cname, {}).get(v, 0))
            f.write(f"  {cname}: [{best_ver}] {line}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
