"""Stage 1: SAM2 image predictor per-frame, with the SAME box prompts each frame.

The camera is rigidly mounted, so the same coarse box prompts that worked on
frame 500 work for every frame. This avoids the O(n^2) memory growth seen in
SAM2's video predictor on long sequences with limited VRAM.

Runs at ~3-5 fps on an 8GB 4070 with the tiny model → ~5 min for 1163 frames.
"""
import os, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "frames_scene_rgb_2"
SMALL_DIR = ROOT / "data" / "frames_scene_rgb_2_small"
OUT_DIR = ROOT / "artifacts" / "masks"
OVERLAY_DIR = ROOT / "artifacts" / "masks_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# Working size + prompts (in 512x512 coords)
NEG_PTS = np.array([(255, 240), (300, 250), (350, 350), (290, 270)], dtype=np.float32)
NEG_LBL = np.array([0]*len(NEG_PTS), dtype=np.int32)
BOXES = {
    "body":  (5, 70, 135, 270),
    "arm_u": (95, 130, 230, 220),
    "arm_l": (95, 200, 240, 290),
}
DILATE_PX = 22   # heavy dilation: grabs the gripper tips touching the mat
                 # without explicitly prompting on them

frame_files = sorted(FRAMES_DIR.glob("*.png"))
small_files = sorted(SMALL_DIR.glob("*.jpg"))
print(f"frames: {len(frame_files)} (small: {len(small_files)})")
assert len(small_files) == len(frame_files), "small frames missing"

print("loading SAM2 image predictor (tiny) ...")
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", device="cuda")
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX*2+1, DILATE_PX*2+1))

t0 = time.time()
n_done = 0

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i, sp in enumerate(small_files):
        img = np.array(Image.open(sp).convert("RGB"))
        predictor.set_image(img)
        # Predict each object's mask independently, then union
        unions = np.zeros((512, 512), dtype=bool)
        for name, box in BOXES.items():
            masks, _, _ = predictor.predict(
                box=np.array(box, dtype=np.float32),
                point_coords=NEG_PTS, point_labels=NEG_LBL,
                multimask_output=False,
            )
            unions |= masks[0].astype(bool)
        # Dilate at 512 first (kernel is sized for 512-space), then upscale
        mask_512 = unions.astype(np.uint8) * 255
        mask_512 = cv2.dilate(mask_512, KERNEL, iterations=1)
        mask_1024 = cv2.resize(mask_512, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT_DIR / f"{i:06d}.png"), mask_1024)
        # Overlay every 50 frames for QC
        if i % 50 == 0:
            rgb = cv2.imread(str(frame_files[i]))
            colored = rgb.copy()
            colored[mask_1024 > 0] = (0.4 * np.array([0, 0, 255]) + 0.6 * colored[mask_1024 > 0]).astype(np.uint8)
            cv2.imwrite(str(OVERLAY_DIR / f"{i:06d}.png"), colored)
        n_done += 1
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {n_done}/{len(small_files)}  {elapsed:.1f}s  ({n_done/elapsed:.2f} fps)", flush=True)

print(f"done. {n_done} masks in {time.time()-t0:.1f}s")
