"""Pipeline v2: keep the original video, overlay tracking markers + a
right-side HUD column of per-arm camera feeds and sensor plots, each
connected to its gripper by a coloured line.

Two outputs per run:
  artifacts/overlay_displays/      — HUD on the original pinhole frames
  artifacts/overlay_displays_yam/  — HUD on the original pinhole frames with
                                     the YAM render chroma-keyed on top

Usage:
  python scripts/33_overlay_displays.py [frame_limit]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import cv2

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all frames

ROOT = Path(__file__).resolve().parent.parent
calib = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
meta  = json.loads((ROOT / "data" / "frames_meta.json").read_text())

K     = np.array(calib["K"], dtype=np.float64)
rvec  = np.array(calib["rvec"], dtype=np.float64)
tvec  = np.array(calib["tvec"], dtype=np.float64)
R_a1_to_a2 = np.array(calib.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(calib.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
CAL_ARM = calib.get("calibrated_arm", 2)

FRAMES_DIR = ROOT / "artifacts" / "pinhole"
RENDER_DIR = ROOT / "artifacts" / "render"          # YAM-only renders
FRAMES = sorted(FRAMES_DIR.glob("*.png"))
if LIMIT > 0:
    FRAMES = FRAMES[:LIMIT]
N = len(FRAMES)
print(f"pinhole frames: {N} (limit arg: {LIMIT})")

OUT_RAW = ROOT / "artifacts" / "overlay_displays"
OUT_YAM = ROOT / "artifacts" / "overlay_displays_yam"
OUT_RAW.mkdir(exist_ok=True, parents=True)
OUT_YAM.mkdir(exist_ok=True, parents=True)

# ----- RealSense streams --------------------------------------------------
VIDEO_BASE = ROOT / "data" / "hf" / "videos"
RS_PATHS = {
    1: {
        "rgb":   VIDEO_BASE / "observation.images.realsense_rgb"       / "chunk-000" / "file-000.mp4",
        "depth": VIDEO_BASE / "observation.images.realsense_rgb_depth" / "chunk-000" / "file-000.mp4",
    },
    2: {
        "rgb":   VIDEO_BASE / "observation.images.arm2_realsense_rgb"       / "chunk-000" / "file-000.mp4",
        "depth": VIDEO_BASE / "observation.images.arm2_realsense_rgb_depth" / "chunk-000" / "file-000.mp4",
    },
}
caps = {arm: {k: cv2.VideoCapture(str(p)) for k, p in d.items()} for arm, d in RS_PATHS.items()}
for arm, d in caps.items():
    for k, c in d.items():
        if not c.isOpened():
            raise RuntimeError(f"failed to open {RS_PATHS[arm][k]}")

def read_rs(arm, kind):
    ok, f = caps[arm][kind].read()
    if not ok:
        return np.zeros((480, 848, 3), np.uint8)
    return f

# ----- Time series --------------------------------------------------------
def extract_series():
    enc1, enc2, gw1, gw2 = [], [], [], []
    for m in meta:
        s = m.get("state")
        if s is None:
            enc1.append(np.nan); enc2.append(np.nan)
            gw1.append(np.nan); gw2.append(np.nan)
        else:
            enc1.append(float(s[0]));  gw1.append(float(s[15]))
            enc2.append(float(s[16])); gw2.append(float(s[31]))
    return (np.array(enc1, np.float64), np.array(enc2, np.float64),
            np.array(gw1,  np.float64), np.array(gw2,  np.float64))

ENC1, ENC2, GW1, GW2 = extract_series()

def y_range(arr, pad=0.15):
    a = arr[np.isfinite(arr)]
    if len(a) < 2: return (-1.0, 1.0)
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    if hi - lo < 1e-6: lo, hi = lo - 1, hi + 1
    span = hi - lo
    return (lo - pad*span, hi + pad*span)

ENC_RANGE = (min(y_range(ENC1)[0], y_range(ENC2)[0]),
             max(y_range(ENC1)[1], y_range(ENC2)[1]))
GW_RANGE  = (min(y_range(GW1)[0],  y_range(GW2)[0]),
             max(y_range(GW1)[1],  y_range(GW2)[1]))

# ----- Pose -> projected 2D ----------------------------------------------
BASE_OFFSET = np.array([0.0, 0.0, -0.0546], dtype=np.float64)

def quat_to_R(q):
    qx, qy, qz, qw = q
    n = (qx*qx+qy*qy+qz*qz+qw*qw) ** 0.5
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])

def gripper_base_pixel(s, vive_arm):
    """Project the gripper base (midpoint between fingertips at the base)
    to 2D. Returns (u, v) or None if pose is invalid / off-screen."""
    if vive_arm == 1:
        pos = np.array(s[7:10]);   q = s[10:14]
    else:
        pos = np.array(s[23:26]);  q = s[26:30]
    if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
       or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
        return None
    R_g = quat_to_R(q)
    if vive_arm != CAL_ARM:
        Rt, tt = (R_a2_to_a1, t_a2_to_a1) if CAL_ARM == 1 else (R_a1_to_a2, t_a1_to_a2)
        pos = Rt @ pos + tt
        R_g = Rt @ R_g
    base_w = pos + R_g @ BASE_OFFSET
    proj, _ = cv2.projectPoints(base_w.reshape(1, 3), rvec, tvec, K, None)
    u, v = proj[0, 0]
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return float(u), float(v)

# ----- HUD layout ---------------------------------------------------------
IMG_W, IMG_H = 1024, 1024
COL_MARGIN_R = 18                         # right margin
PANEL_W      = 280                        # column width
COL_RIGHT    = IMG_W - COL_MARGIN_R       # 1006
COL_LEFT     = COL_RIGHT - PANEL_W        # 726

CAM_H, SENSE_H = 136, 96                  # panel heights
ARM_BLOCK_H    = CAM_H + 6 + SENSE_H      # cam + small gap + sense
INTER_ARM_GAP  = 28
TOTAL_H        = ARM_BLOCK_H * 2 + INTER_ARM_GAP
TOP_Y          = (IMG_H - TOTAL_H) // 2

def arm_panel_rects(arm):
    """Return (cam_rect, sense_rect) as (x0, y0, x1, y1) for the given arm.
    Arm 1 is the top block, arm 2 the bottom block."""
    block_top = TOP_Y + (0 if arm == 1 else ARM_BLOCK_H + INTER_ARM_GAP)
    cam = (COL_LEFT, block_top,                    COL_RIGHT, block_top + CAM_H)
    sense_top = block_top + CAM_H + 6
    sense = (COL_LEFT, sense_top, COL_RIGHT, sense_top + SENSE_H)
    return cam, sense

ARM_COL   = {1: (0, 220, 255), 2: (255, 200, 0)}
ARM_LABEL = {1: "ARM 1", 2: "ARM 2"}

# ----- Panel drawing helpers ---------------------------------------------
def draw_cam_panel(img, rect, rgb_frame, depth_frame, label, col):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    half_w = w // 2
    inner_h = h - 22
    rgb_small   = cv2.resize(rgb_frame,   (half_w, inner_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_frame, (w - half_w, inner_h), interpolation=cv2.INTER_AREA)
    # Title strip
    cv2.rectangle(img, (x0, y0), (x1, y0 + 22), (28, 28, 28), -1)
    cv2.putText(img, f"{label} CAM", (x0 + 8, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    img[y0 + 22:y0 + 22 + inner_h, x0:x0 + half_w] = rgb_small
    img[y0 + 22:y0 + 22 + inner_h, x0 + half_w:x1] = depth_small
    cv2.putText(img, "RGB",   (x0 + 6, y0 + 22 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "DEPTH", (x0 + half_w + 6, y0 + 22 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (x0 + half_w, y0 + 22), (x0 + half_w, y1), (60, 60, 60), 1)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, 2)

def draw_sense_panel(img, rect, i, gw_series, enc_series, label, col, window=120):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    cv2.rectangle(img, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.rectangle(img, (x0, y0), (x1, y0 + 20), (28, 28, 28), -1)
    cv2.putText(img, f"{label} STATE", (x0 + 8, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    plot_top, plot_bot = y0 + 24, y1 - 4
    plot_left, plot_right = x0 + 36, x1 - 6
    pw = plot_right - plot_left
    ph = plot_bot - plot_top
    cv2.rectangle(img, (plot_left, plot_top), (plot_right, plot_bot),
                  (60, 60, 60), 1)
    half_ph = ph // 2
    cv2.line(img, (plot_left, plot_top + half_ph),
             (plot_right, plot_top + half_ph), (40, 40, 40), 1)

    lo_idx = max(0, i - window + 1)
    xs = np.arange(lo_idx, i + 1)
    def to_pts(values, vrange, top, bot):
        vlo, vhi = vrange
        if vhi <= vlo: vhi = vlo + 1
        u = (xs - (i - window + 1)) / max(1, window - 1)
        u = np.clip(u, 0.0, 1.0)
        x = plot_left + u * pw
        v = (values - vlo) / (vhi - vlo)
        v = np.clip(v, 0.0, 1.0)
        y = bot - v * (bot - top)
        return x, y
    band1 = (plot_top + 2,         plot_top + half_ph - 2)
    band2 = (plot_top + half_ph,   plot_bot - 2)

    vals_gw = gw_series[lo_idx:i+1]
    x, y = to_pts(vals_gw, GW_RANGE, *band1)
    finite = np.isfinite(vals_gw)
    pts = np.stack([x[finite], y[finite]], axis=1).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, (220, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "gw", (x0 + 4, band1[0] + 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (220, 220, 0), 1, cv2.LINE_AA)

    vals_enc = enc_series[lo_idx:i+1]
    x, y = to_pts(vals_enc, ENC_RANGE, *band2)
    finite = np.isfinite(vals_enc)
    pts = np.stack([x[finite], y[finite]], axis=1).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, (60, 160, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "enc", (x0 + 4, band2[0] + 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (60, 160, 255), 1, cv2.LINE_AA)

    cv2.line(img, (plot_right, plot_top), (plot_right, plot_bot),
             (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, 2)

def draw_connector(img, panel_rect, gripper_uv, col):
    """Thin line from the panel's left-middle edge to the gripper pixel."""
    x0, y0, x1, y1 = panel_rect
    px = x0
    py = (y0 + y1) // 2
    gu, gv = int(round(gripper_uv[0])), int(round(gripper_uv[1]));
    cv2.line(img, (px, py), (gu, gv), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (px, py), (gu, gv), col, 1, cv2.LINE_AA)
    # Anchor dot on the panel
    cv2.circle(img, (px, py), 4, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 3, col, -1, cv2.LINE_AA)

def draw_marker(img, gripper_uv, arm, col):
    u, v = int(round(gripper_uv[0])), int(round(gripper_uv[1]))
    cv2.circle(img, (u, v), 14, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), 12, col, -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, f"a{arm}", (u + 16, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

def draw_hud(img, i, s, rs):
    """Mutates `img` in place: draws connectors, panels, markers."""
    gripper_uv = {a: (gripper_base_pixel(s, a) if s is not None else None)
                  for a in (1, 2)}

    # Draw connectors first so they sit under the panels
    for arm in (1, 2):
        if gripper_uv[arm] is None: continue
        cam_rect, sense_rect = arm_panel_rects(arm)
        col = ARM_COL[arm]
        draw_connector(img, cam_rect,   gripper_uv[arm], col)
        draw_connector(img, sense_rect, gripper_uv[arm], col)

    # Draw panels
    for arm in (1, 2):
        cam_rect, sense_rect = arm_panel_rects(arm)
        col = ARM_COL[arm]
        draw_cam_panel(img, cam_rect, rs[arm]["rgb"], rs[arm]["depth"],
                       ARM_LABEL[arm], col)
        gw_s  = GW1  if arm == 1 else GW2
        enc_s = ENC1 if arm == 1 else ENC2
        draw_sense_panel(img, sense_rect, i, gw_s, enc_s,
                         ARM_LABEL[arm], col)

    # Markers on top
    for arm in (1, 2):
        if gripper_uv[arm] is None: continue
        draw_marker(img, gripper_uv[arm], arm, ARM_COL[arm])

# ----- YAM chroma-key composite ------------------------------------------
def composite_yam(bg, yam):
    """Luminance chroma-key composite, matching 15_composite_pinhole.py."""
    if bg.shape != yam.shape:
        yam = cv2.resize(yam, (bg.shape[1], bg.shape[0]))
    luma  = cv2.cvtColor(yam, cv2.COLOR_BGR2GRAY)
    mask  = (luma > 8).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    a = np.stack([alpha] * 3, axis=-1)
    return (a * yam + (1 - a) * bg).astype(np.uint8)

# ----- Main loop ---------------------------------------------------------
t0 = time.time()
yam_frames = sorted(RENDER_DIR.glob("*.png"))
have_yam = len(yam_frames) >= N
if not have_yam:
    print(f"warning: render/ has {len(yam_frames)} frames, expected >= {N}; "
          f"skipping YAM variant")

for i, fp in enumerate(FRAMES):
    img_raw = cv2.imread(str(fp))
    img_yam = img_raw.copy() if have_yam else None
    if have_yam:
        yam = cv2.imread(str(yam_frames[i]))
        img_yam = composite_yam(img_yam, yam)

    rs = {arm: {k: read_rs(arm, k) for k in ("rgb", "depth")} for arm in (1, 2)}
    s = meta[i].get("state")

    draw_hud(img_raw, i, s, rs)
    cv2.imwrite(str(OUT_RAW / f"{i:06d}.png"), img_raw)

    if have_yam:
        draw_hud(img_yam, i, s, rs)
        cv2.imwrite(str(OUT_YAM / f"{i:06d}.png"), img_yam)

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N}  {time.time()-t0:.1f}s")

for arm, d in caps.items():
    for c in d.values():
        c.release()

print(f"done in {time.time()-t0:.1f}s, wrote {N} frames")
print(f"  raw -> {OUT_RAW}")
if have_yam:
    print(f"  yam -> {OUT_YAM}")
