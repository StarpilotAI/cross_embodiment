"""Composite the debug-render (axes + cube + markers + trails) onto the
pinhole-undistorted dataset frames (NOT the inpainted ones — we want to see
the operator's actual gripper position alongside the rendered markers).
"""
import time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
BG = sorted((ROOT / "artifacts" / "pinhole").glob("*.png"))
FG = sorted((ROOT / "artifacts" / "debug_render").glob("*.png"))
OUT = ROOT / "artifacts" / "debug_composite"
OUT.mkdir(parents=True, exist_ok=True)

n = min(len(BG), len(FG))
print(f"compositing {n} debug frames over pinhole bg")
t0 = time.time()
for i in range(n):
    bg = cv2.imread(str(BG[i]))
    fg = cv2.imread(str(FG[i]))
    if bg.shape != fg.shape:
        fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]))
    luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    mask = (luma > 8).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(mask, (3,3), 0).astype(np.float32) / 255.0
    a = np.stack([alpha]*3, axis=-1)
    comp = (a * fg + (1-a) * bg).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{i:06d}.png"), comp)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{n}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s")
