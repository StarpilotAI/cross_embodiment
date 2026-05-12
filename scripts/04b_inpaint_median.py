"""Fast background-fill inpainter that exploits a fixed camera.

For each pixel position, we collect RGB samples from frames where the mask
is zero (= pixel is "background" in that frame). We take the median across
those samples to build a clean per-pixel "empty workstation" plate. Then
each frame's masked region is filled from that plate.

This is dramatically faster than ProPainter (~30s vs 15min for 1163 frames)
and works very well when:
  - the camera is rigid
  - the masked region is occupied by background at least sometimes
For this dataset (fixed fisheye + person mostly on left), that's true for the
table+mat area but NOT for the area always covered by the person.

We blend the result with the original frame to keep mat motion clean.
"""
import time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
FRAMES = sorted((ROOT / "data" / "frames_scene_rgb_2").glob("*.png"))
MASKS  = sorted((ROOT / "artifacts" / "masks").glob("*.png"))
OUT = ROOT / "artifacts" / "inpainted"
OUT.mkdir(parents=True, exist_ok=True)

assert len(FRAMES) == len(MASKS) and len(FRAMES) > 0
print(f"frames: {len(FRAMES)}")

# --- Build per-pixel median background by subsampling 100 frames ---
SUB = 12  # use every Nth frame for the median plate
sub_idx = list(range(0, len(FRAMES), SUB))
print(f"computing median background from {len(sub_idx)} sample frames...")
t0 = time.time()
N = len(sub_idx)
H = W = 1024
# stack of "valid" samples per pixel. We use a fixed-size sample pool of N frames,
# where masked pixels are set to NaN and then median ignores NaN. Float32 = 4
# bytes × 1024×1024×3×N — for N=97 that's ~1.1 GB. OK on 16+ GB RAM.
buf = np.empty((N, H, W, 3), dtype=np.float32)
for k, i in enumerate(sub_idx):
    img = cv2.imread(str(FRAMES[i]))
    mask = cv2.imread(str(MASKS[i]), cv2.IMREAD_GRAYSCALE)
    img_f = img.astype(np.float32)
    img_f[mask > 0] = np.nan
    buf[k] = img_f
print(f"  buf built in {time.time()-t0:.1f}s, shape={buf.shape}")
# Median ignoring NaN. Pixels that are ALWAYS masked → NaN; we'll fall back to
# a nearby-frame value.
t0 = time.time()
median_bg = np.nanmedian(buf, axis=0)  # (H,W,3)
del buf  # free RAM
print(f"  nanmedian in {time.time()-t0:.1f}s")
# Fill any remaining NaN with the mean of valid neighbors (or a constant)
nan_mask = np.isnan(median_bg).any(axis=-1)
print(f"  pixels with no valid background sample: {int(nan_mask.sum())}/{H*W}")
if nan_mask.any():
    # Fill no-valid-sample pixels with BLACK — these are areas where the operator
    # is ALWAYS present (left side of the fisheye, overlapping the dark tent
    # canopy). Black matches the scene more faithfully than the TELEA fallback
    # which would copy mat-coloured pixels and produce a "yellow blob".
    median_bg = np.nan_to_num(median_bg, nan=0.0)
median_bg_u8 = median_bg.astype(np.uint8)
cv2.imwrite(str(ROOT / "artifacts" / "median_background.png"), median_bg_u8)
print(f"  saved median plate to artifacts/median_background.png")

# --- Per-frame composite: original where mask=0, median_bg where mask=1 ---
print("compositing per-frame inpaint...")
t0 = time.time()
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
for i, (fp, mp) in enumerate(zip(FRAMES, MASKS)):
    img = cv2.imread(str(fp))
    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    # Feather mask edges to hide seams
    mask_feathered = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
    alpha = np.stack([mask_feathered]*3, axis=-1)
    comp = (alpha * median_bg_u8 + (1 - alpha) * img).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{i:06d}.png"), comp)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(FRAMES)}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s. wrote {len(FRAMES)} inpainted frames")
