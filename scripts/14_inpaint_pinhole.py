"""Stage 4 (new pipeline): LaMa inpainting on PINHOLE frames + pinhole masks.

Output: artifacts/pinhole_inpainted/*.png
"""
import time
from pathlib import Path
from PIL import Image
from simple_lama_inpainting import SimpleLama

ROOT = Path(__file__).resolve().parent.parent
FRAMES = sorted((ROOT / "artifacts" / "pinhole").glob("*.png"))
MASKS  = sorted((ROOT / "artifacts" / "pinhole_masks").glob("*.png"))
OUT = ROOT / "artifacts" / "pinhole_inpainted"
OUT.mkdir(parents=True, exist_ok=True)

print(f"pinhole frames: {len(FRAMES)}, masks: {len(MASKS)}")
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
