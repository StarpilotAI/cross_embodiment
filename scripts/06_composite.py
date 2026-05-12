"""Stage 5: depth-aware composite of YAM render onto inpainted background.

For each pixel:
  - background = inpainted scene_rgb_2
  - foreground = YAM render
  - alpha = (render is non-background?) AND (render_depth < scene_depth)

Right now (tier-1) we approximate the depth z-test with a hard mask: pixels
where the YAM render is non-empty get overlaid. The Depth Anything V2 scene
depth is shown in the viewer but not used in the composite, because the YAM
render is currently in its own camera frame, not aligned with the scene cam.
This becomes meaningful once scene-cam extrinsics are calibrated.
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
INPAINT_DIR = ROOT / "artifacts" / "inpainted"
RENDER_DIR = ROOT / "artifacts" / "render"
OUT_DIR = ROOT / "artifacts" / "composite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the original frames as background if inpaint isn't ready
if not list(INPAINT_DIR.glob("*.png")):
    INPAINT_DIR = ROOT / "data" / "frames_scene_rgb_2"
    print("inpaint not ready, using ORIGINAL frames as background")

bg_files = sorted(INPAINT_DIR.glob("*.png"))
fg_files = sorted(RENDER_DIR.glob("*.png"))
n = min(len(bg_files), len(fg_files))
print(f"bg={len(bg_files)} fg={len(fg_files)}  compositing {n} frames")

for i in range(n):
    bg = cv2.imread(str(bg_files[i]))
    fg = cv2.imread(str(fg_files[i]))
    if bg.shape != fg.shape:
        fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]))

    # Render uses a pure-black background; anything brighter is robot.
    luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    fg_mask = (luma > 8).astype(np.uint8) * 255

    # Smooth alpha
    alpha = cv2.GaussianBlur(fg_mask, (5, 5), 0).astype(np.float32) / 255.0
    alpha = np.stack([alpha]*3, axis=-1)
    comp = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / f"{i:06d}.png"), comp)
    if i % 200 == 0:
        print(f"  {i}/{n}")

print("done")
