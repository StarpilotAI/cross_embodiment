"""Stage 2: ProPainter video inpainting.

We process at 512x512 (matches SAM2 mask resolution and fits in 8GB VRAM).
The output is then upscaled back to 1024 for downstream stages.

ProPainter writes its result mp4 + (optionally) frame PNGs into a subdir of
the output folder named after the input video. Here we pass a frame folder
so the subdir is named after the folder.
"""
import os, sys, shutil, subprocess
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
PROPAINTER = ROOT / "data" / "ProPainter"
FRAMES_SMALL = ROOT / "data" / "frames_scene_rgb_2_small_jpg512"
MASKS_SMALL  = ROOT / "data" / "masks_small512"
OUT = ROOT / "artifacts" / "inpaint_raw"
FINAL_DIR = ROOT / "artifacts" / "inpainted"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ProPainter wants both frames and masks at the same resolution and in
# directories with matching filenames. Let's prep 512 mask PNGs that share
# the frame numbering.
src_frames = sorted((ROOT / "data" / "frames_scene_rgb_2").glob("*.png"))
src_masks  = sorted((ROOT / "artifacts" / "masks").glob("*.png"))
print(f"frames: {len(src_frames)}, masks: {len(src_masks)}")
assert len(src_masks) > 0, "run scripts/02_mask.py first"

FRAMES_SMALL.mkdir(parents=True, exist_ok=True)
MASKS_SMALL.mkdir(parents=True, exist_ok=True)
if len(list(FRAMES_SMALL.glob("*.png"))) != len(src_frames):
    for fp in src_frames:
        img = cv2.imread(str(fp))
        cv2.imwrite(str(FRAMES_SMALL / fp.name), cv2.resize(img, (512, 512), cv2.INTER_AREA))
if len(list(MASKS_SMALL.glob("*.png"))) != len(src_masks):
    for fp in src_masks:
        m = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(MASKS_SMALL / fp.name), cv2.resize(m, (512, 512), cv2.INTER_NEAREST))
print(f"  prepped {len(list(FRAMES_SMALL.glob('*.png')))} 512x512 frame+mask pairs")

# Run ProPainter
cmd = [
    sys.executable, "inference_propainter.py",
    "-i", str(FRAMES_SMALL),
    "-m", str(MASKS_SMALL),
    "-o", str(OUT),
    "--fp16",
    "--save_frames",
    "--subvideo_length", "40",  # short to fit in 8GB VRAM
    "--neighbor_length", "10",
    "--raft_iter", "10",
    "--save_fps", "30",
]
print("running propainter:", " ".join(cmd))
subprocess.run(cmd, cwd=str(PROPAINTER), check=True)

# ProPainter saves under OUT / video_name / frames / *.png
result_dir = OUT / FRAMES_SMALL.name / "frames"
out_files = sorted(result_dir.glob("*.png"))
print(f"propainter wrote {len(out_files)} frames to {result_dir}")

# Copy back to FINAL_DIR with renamed filenames (frame number)
for i, fp in enumerate(out_files):
    img = cv2.imread(str(fp))
    img1024 = cv2.resize(img, (1024, 1024), cv2.INTER_CUBIC)
    cv2.imwrite(str(FINAL_DIR / f"{i:06d}.png"), img1024)
print(f"upscaled + saved to {FINAL_DIR}")
