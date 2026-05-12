"""Sanity-check SAM2 prompts on pinhole frames."""
import numpy as np, cv2, torch
from pathlib import Path
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parent.parent
INSPECT = ROOT / "data" / "inspect"

# Frames are 1024x1024 pinhole.  Working at 512 for SAM2 speed.
boxes_named = {
    "person":   (0, 0, 140, 350),
    "arm":      (0, 260, 240, 500),
}
# Strong negatives on the mat (centre-right) to prevent bleed
neg_pts = [(220, 200), (280, 250), (320, 300), (350, 350), (200, 300)]

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", device="cuda")

for anchor in (100, 250, 500, 750, 1000):
    src = ROOT / "artifacts" / "pinhole" / f"{anchor:06d}.png"
    img_full = cv2.imread(str(src))
    img = cv2.resize(img_full, (512, 512), cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img.copy()
    print(f"\nframe {anchor}:")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img_rgb)
        for name, box in boxes_named.items():
            masks, scores, _ = predictor.predict(
                box=np.array(box, dtype=np.float32),
                point_coords=np.array(neg_pts, dtype=np.float32),
                point_labels=np.array([0]*len(neg_pts), dtype=np.int32),
                multimask_output=False,
            )
            m = masks[0].astype(bool)
            c = np.array([0,0,255] if name=="person" else [0,255,0], dtype=np.uint8)
            overlay[m] = (0.45*overlay[m] + 0.55*c).astype(np.uint8)
            x1,y1,x2,y2 = box
            cv2.rectangle(overlay, (x1,y1), (x2,y2), c.tolist(), 1)
            print(f"  {name:6s} pix={int(m.sum())} score={float(scores[0]):.3f}")
        for pt in neg_pts: cv2.circle(overlay, pt, 3, (255,255,255), -1)
    cv2.imwrite(str(INSPECT / f"pin_sanity_{anchor}.png"), overlay)
print("wrote pin_sanity_*.png")
