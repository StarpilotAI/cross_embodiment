"""Stage 3: Depth Anything V2 monocular depth on scene_rgb_2 frames.

Saves a relative-depth 16-bit PNG per frame to artifacts/depth/, plus an
8-bit colormap preview to artifacts/depth_preview/.

We use the "Small" variant for VRAM headroom (the dataset is 1163 frames).
The depth is *not* metric out of the box; we use it relatively for the
depth-aware compositing z-test (robot vs scene depth).
"""
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from transformers import pipeline
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "frames_scene_rgb_2"
OUT_DIR = ROOT / "artifacts" / "depth"
PREVIEW_DIR = ROOT / "artifacts" / "depth_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

print("loading Depth Anything V2 small ...")
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0,
    torch_dtype=torch.float16,
)
print("  ready")

frames = sorted(FRAMES_DIR.glob("*.png"))
print(f"frames: {len(frames)}")

t0 = time.time()
for i, fp in enumerate(frames):
    img = Image.open(fp).convert("RGB")
    out = pipe(img)
    depth = out["predicted_depth"][0].cpu().float().numpy()  # H,W float32
    # Resize to source resolution
    depth = cv2.resize(depth, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # Save raw float as 16-bit PNG (rescaled to [0, 65535] using THIS frame's range)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        d16 = np.zeros_like(depth, dtype=np.uint16)
    else:
        d16 = ((depth - dmin) / (dmax - dmin) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(OUT_DIR / f"{i:06d}.png"), d16)

    # Preview colormap every 10 frames
    if i % 10 == 0:
        d8 = ((depth - dmin) / (dmax - dmin + 1e-9) * 255).astype(np.uint8)
        cm = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(PREVIEW_DIR / f"{i:06d}.png"), cm)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{len(frames)}  {elapsed:.1f}s  ({(i+1)/elapsed:.1f} fps)")

print(f"done. {len(frames)} depth frames in {time.time()-t0:.1f}s")
