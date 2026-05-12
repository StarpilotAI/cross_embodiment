"""Stage 3 (improved): Depth Anything V2 on the undistorted pinhole frames.

Monocular depth works much better on normal pinhole imagery than on fisheye —
the model is in-distribution. We save 16-bit relative depth + a turbo
colormap preview.
"""
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from transformers import pipeline
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "artifacts" / "pinhole"
OUT_DIR = ROOT / "artifacts" / "depth_pinhole"
PREVIEW_DIR = ROOT / "artifacts" / "depth_pinhole_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

print("loading Depth Anything V2 small ...")
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0,
    torch_dtype=torch.float16,
)

frames = sorted(FRAMES_DIR.glob("*.png"))
print(f"pinhole frames: {len(frames)}")
assert frames, "run scripts/07_undistort.py first"

t0 = time.time()
for i, fp in enumerate(frames):
    img = Image.open(fp).convert("RGB")
    out = pipe(img)
    depth = out["predicted_depth"][0].cpu().float().numpy()
    depth = cv2.resize(depth, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        d16 = np.zeros_like(depth, dtype=np.uint16)
    else:
        d16 = ((depth - dmin) / (dmax - dmin) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(OUT_DIR / f"{i:06d}.png"), d16)
    if i % 10 == 0:
        d8 = ((depth - dmin) / (dmax - dmin + 1e-9) * 255).astype(np.uint8)
        cm = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(PREVIEW_DIR / f"{i:06d}.png"), cm)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(frames)}  {time.time()-t0:.1f}s", flush=True)

print(f"done in {time.time()-t0:.1f}s")
