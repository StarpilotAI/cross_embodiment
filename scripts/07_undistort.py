"""Fisheye undistortion → pinhole crop for downstream depth estimation.

We don't have explicit fisheye intrinsics, so we ESTIMATE them assuming an
equidistant fisheye (r = f * theta) that fills the 1024x1024 image:
  - principal point  = (512, 512)
  - disk radius      ~ 510 px (visible bright disk in the dataset previews)
  - max view angle   ~ pi/2  (90deg half-FOV  i.e. 180deg full)
  - focal length     ~ radius / (pi/2)   ~ 325 px

The undistorted target is a pinhole crop covering the central workspace
(~60deg horizontal FOV). For a 1024x1024 target with FOV=60deg:
  fx = fy = 1024 / (2 * tan(30deg)) ~ 886
"""
import time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
FRAMES = sorted((ROOT / "data" / "frames_scene_rgb_2").glob("*.png"))
MASKS  = sorted((ROOT / "artifacts" / "masks").glob("*.png"))
OUT_RGB  = ROOT / "artifacts" / "pinhole"
OUT_MASK = ROOT / "artifacts" / "pinhole_mask"
OUT_RGB.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

H = W = 1024
# Fisheye intrinsics (equidistant guess)
F_FISH = 325.0
K_fish = np.array([[F_FISH, 0, W/2],
                   [0, F_FISH, H/2],
                   [0,      0,   1]], dtype=np.float64)
D_fish = np.zeros(4, dtype=np.float64)

# Pinhole target: 60deg horizontal FOV centered on workspace
FOV_DEG = 60
F_PIN = (W/2) / np.tan(np.deg2rad(FOV_DEG/2))
K_pin = np.array([[F_PIN, 0, W/2],
                  [0, F_PIN, H/2],
                  [0,    0,   1]], dtype=np.float64)

# Build the undistortion map once
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K_fish, D_fish, np.eye(3), K_pin, (W, H), cv2.CV_16SC2)

t0 = time.time()
for i, (fp, mp) in enumerate(zip(FRAMES, MASKS)):
    img = cv2.imread(str(fp))
    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    pin_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    pin_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(OUT_RGB / f"{i:06d}.png"), pin_img)
    cv2.imwrite(str(OUT_MASK / f"{i:06d}.png"), pin_mask)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(FRAMES)}  {time.time()-t0:.1f}s", flush=True)

print(f"done in {time.time()-t0:.1f}s")
