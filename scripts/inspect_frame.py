"""Save an annotated version of frame 0 at 512 (and a couple later frames)
with a 32px grid overlay so we can pick click coordinates."""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMALL_DIR = ROOT / "data" / "frames_scene_rgb_2_small"
OUT_DIR = ROOT / "data" / "inspect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for idx in (0, 200, 500, 800, 1100):
    p = SMALL_DIR / f"{idx:05d}.jpg"
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    # Light grid every 32 px
    for x in range(0, w, 32):
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(img, str(x), (x+1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    for y in range(0, h, 32):
        cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, str(y), (1, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    cv2.imwrite(str(OUT_DIR / f"grid_{idx:05d}.png"), img)
    print(f"  wrote {OUT_DIR / f'grid_{idx:05d}.png'}")
