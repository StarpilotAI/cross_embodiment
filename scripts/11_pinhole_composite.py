"""Undistort the LaMa-inpainted fisheye frames to pinhole, then composite the
calibrated YAM render on top. Output: artifacts/composite_pinhole/*.png.

Uses the SAME fisheye→pinhole map as scripts/07_undistort.py, so the
calibrated camera (which is the pinhole intrinsics K_PIN) aligns naturally
with the undistorted background.
"""
import time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
FRAMES_INP = sorted((ROOT / "artifacts" / "inpainted").glob("*.png"))
FRAMES_RENDER = sorted((ROOT / "artifacts" / "render").glob("*.png"))
OUT = ROOT / "artifacts" / "composite_pinhole"
OUT.mkdir(parents=True, exist_ok=True)

# Same as scripts/07_undistort.py
W = H = 1024
F_FISH = 325.0
K_FISH = np.array([[F_FISH, 0, W/2], [0, F_FISH, H/2], [0, 0, 1]], dtype=np.float64)
D_FISH = np.zeros(4, dtype=np.float64)
F_PIN = (W/2) / np.tan(np.deg2rad(60/2))
K_PIN = np.array([[F_PIN, 0, W/2], [0, F_PIN, H/2], [0, 0, 1]], dtype=np.float64)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_FISH, D_FISH, np.eye(3), K_PIN, (W, H), cv2.CV_16SC2)

n = min(len(FRAMES_INP), len(FRAMES_RENDER))
print(f"compositing {n} frames (pinhole)")
t0 = time.time()
for i in range(n):
    bg_fish = cv2.imread(str(FRAMES_INP[i]))
    bg_pin = cv2.remap(bg_fish, map1, map2, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    fg = cv2.imread(str(FRAMES_RENDER[i]))
    # Render uses black bg; chroma-key on luminance
    luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    mask = (luma > 8).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(mask, (5,5), 0).astype(np.float32) / 255.0
    a = np.stack([alpha]*3, axis=-1)
    comp = (a * fg + (1-a) * bg_pin).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{i:06d}.png"), comp)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{n}  {time.time()-t0:.1f}s", flush=True)
print(f"done in {time.time()-t0:.1f}s")
