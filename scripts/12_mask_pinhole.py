"""Stage 3 (new pipeline): SAM2 mask of (person + arm + gripper) on PINHOLE frames.

Camera is static so the SAME bbox prompts work for every frame. We operate at
512x512 internally (much faster) then upscale masks to 1024x1024 with heavy
dilation so the mask thoroughly engulfs the gripper tips touching the mat.

Output: artifacts/pinhole_masks/*.png  (binary, 255 = remove)
"""
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "artifacts" / "pinhole"
OUT_DIR = ROOT / "artifacts" / "pinhole_masks"
OVERLAY_DIR = ROOT / "artifacts" / "pinhole_masks_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# 512-space prompts (verified in pin_sanity_*)
NEG_PTS = np.array([(220, 200), (280, 250), (320, 300), (350, 350), (200, 300)],
                   dtype=np.float32)
NEG_LBL = np.array([0]*len(NEG_PTS), dtype=np.int32)
BOXES = {
    "person": (0, 0, 140, 350),
    "arm":    (0, 260, 240, 500),
}
DILATE_PX = 18

frames = sorted(FRAMES_DIR.glob("*.png"))
print(f"pinhole frames: {len(frames)}")
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", device="cuda")
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX*2+1, DILATE_PX*2+1))

t0 = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i, fp in enumerate(frames):
        full = cv2.imread(str(fp))
        small = cv2.resize(full, (512, 512), cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        union = np.zeros((512, 512), dtype=bool)
        for name, box in BOXES.items():
            masks, _, _ = predictor.predict(
                box=np.array(box, dtype=np.float32),
                point_coords=NEG_PTS, point_labels=NEG_LBL,
                multimask_output=False,
            )
            union |= masks[0].astype(bool)
        mask_512 = union.astype(np.uint8) * 255
        mask_512 = cv2.dilate(mask_512, KERNEL, iterations=1)
        mask_1024 = cv2.resize(mask_512, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT_DIR / f"{i:06d}.png"), mask_1024)
        if i % 50 == 0:
            ov = full.copy()
            ov[mask_1024 > 0] = (0.4*np.array([0,0,255]) + 0.6*ov[mask_1024 > 0]).astype(np.uint8)
            cv2.imwrite(str(OVERLAY_DIR / f"{i:06d}.png"), ov)
        if (i+1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(frames)}  {el:.1f}s  ({(i+1)/el:.2f} fps)", flush=True)

print(f"done. {len(frames)} masks in {time.time()-t0:.1f}s")
