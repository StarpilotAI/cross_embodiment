"""Stage 2 (proper): LaMa video inpainting per-frame.

LaMa is the standard image inpainter (Suvorov et al. 2022). Runs on GPU
at ~3-5 fps for 1024x1024, so ~5 min for 1163 frames. Outputs realistic
fills for the masked person/arm region.
"""
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from simple_lama_inpainting import SimpleLama

ROOT = Path(__file__).resolve().parent.parent
FRAMES = sorted((ROOT / "data" / "frames_scene_rgb_2").glob("*.png"))
MASKS  = sorted((ROOT / "artifacts" / "masks").glob("*.png"))
OUT = ROOT / "artifacts" / "inpainted"
OUT.mkdir(parents=True, exist_ok=True)

print(f"frames: {len(FRAMES)}, masks: {len(MASKS)}")
assert len(FRAMES) == len(MASKS) and FRAMES

print("loading LaMa ...")
lama = SimpleLama()
print("  ready")

t0 = time.time()
for i, (fp, mp) in enumerate(zip(FRAMES, MASKS)):
    img = Image.open(fp).convert("RGB")
    mask = Image.open(mp).convert("L")
    out = lama(img, mask)
    out.save(str(OUT / f"{i:06d}.png"))
    if (i+1) % 50 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(FRAMES)}  {el:.1f}s  ({(i+1)/el:.2f} fps)", flush=True)
print(f"done in {time.time()-t0:.1f}s")
