"""Sanity check: original 3 prompts but with heavy dilation."""
import numpy as np, cv2, torch
from pathlib import Path
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parent.parent
INSPECT = ROOT / "data" / "inspect"

boxes_named = {
    "body":   (5, 70, 135, 270),
    "arm_u":  (95, 130, 230, 220),   # extend right (was 215)
    "arm_l":  (95, 200, 240, 290),
}
neg_pts = [(255, 240), (300, 250), (350, 350), (290, 270)]

DILATE_PX = 22  # heavy dilation to engulf gripper tips touching mat

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", device="cuda")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX*2+1, DILATE_PX*2+1))

for anchor in (200, 500, 700, 800, 1100):
    img_p = ROOT / "data" / "frames_scene_rgb_2_small" / f"{anchor:05d}.jpg"
    img = np.array(Image.open(img_p).convert("RGB"))
    union = np.zeros((512, 512), dtype=bool)
    print(f"\nframe {anchor}:")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img)
        for name, box in boxes_named.items():
            masks, scores, _ = predictor.predict(
                box=np.array(box, dtype=np.float32),
                point_coords=np.array(neg_pts, dtype=np.float32),
                point_labels=np.array([0]*len(neg_pts), dtype=np.int32),
                multimask_output=False,
            )
            union |= masks[0].astype(bool)
            print(f"  {name:6s} pix={int(masks[0].sum())} score={float(scores[0]):.3f}")
    raw = (union.astype(np.uint8) * 255)
    dilated = cv2.dilate(raw, kernel)
    overlay = img.copy()
    overlay[dilated > 0] = (0.45 * overlay[dilated > 0] + 0.55 * np.array([0, 0, 255], np.uint8)).astype(np.uint8)
    for name, box in boxes_named.items():
        x1,y1,x2,y2 = box
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,255), 1)
    cv2.imwrite(str(INSPECT / f"sanity4_{anchor}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("wrote sanity4_*.png")
