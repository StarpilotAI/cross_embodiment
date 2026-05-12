"""Per-frame projection overlay.

For every frame in the episode, project the calibrated Vive gripper pose
through the PnP-calibrated camera and draw a prominent marker on the
pinhole-undistorted dataset image. arm1's pose is transformed to arm2's
frame via the recovered arm1->arm2 rigid transform before projection.

Output: artifacts/proj_overlay/*.png  (every frame, not just clicked ones)
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

ROOT = Path(__file__).resolve().parent.parent
calib = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
meta  = json.loads((ROOT / "data" / "frames_meta.json").read_text())
clicks = json.loads((ROOT / "data" / "calibration_clicks.json").read_text())["clicks"]

K     = np.array(calib["K"], dtype=np.float64)
rvec  = np.array(calib["rvec"], dtype=np.float64)
tvec  = np.array(calib["tvec"], dtype=np.float64)
R_a1_to_a2 = np.array(calib.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(calib.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
CAL_ARM   = calib.get("calibrated_arm", 2)
# Inverse transform (arm2 -> arm1)
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
print(f"calibrated_arm = {CAL_ARM}")

# `base` landmark in gripper local frame (midpoint between fingers at base)
BASE_OFFSET = np.array([0.0, 0.0, -0.0546], dtype=np.float64)

# Gripper geometry for fingertip projection
GRIPPER_XML = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "linear_4310.xml"
grip_m = MjModel.from_xml_path(str(GRIPPER_XML))
grip_d = MjData(grip_m)
TIP_GEOM = {}
for n in ("tip_left", "tip_right"):
    for i in range(grip_m.ngeom):
        if mujoco.mj_id2name(grip_m, mujoco.mjtObj.mjOBJ_BODY, grip_m.geom_bodyid[i]) == n:
            TIP_GEOM[n] = i

def tip_offsets(grip_w):
    half = float(np.clip(grip_w * 0.5, 0.0, 0.0475))
    grip_d.qpos[:] = [half, half]
    mujoco.mj_forward(grip_m, grip_d)
    return (np.asarray(grip_d.geom_xpos[TIP_GEOM["tip_left"]], dtype=np.float64).copy(),
            np.asarray(grip_d.geom_xpos[TIP_GEOM["tip_right"]], dtype=np.float64).copy())

def quat_to_R(q):
    qx, qy, qz, qw = q
    n = (qx*qx+qy*qy+qz*qz+qw*qw) ** 0.5
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])

# Pre-index clicks by frame
clicks_by_frame = {}
for c in clicks:
    clicks_by_frame.setdefault(c["frame"], []).append(c)

OUT = ROOT / "artifacts" / "proj_overlay"
OUT.mkdir(exist_ok=True, parents=True)

frames = sorted((ROOT / "artifacts" / "pinhole").glob("*.png"))
print(f"frames: {len(frames)}")

# Colour palette
COL_BASE_A1 = (0, 220, 255)   # cyan-yellow for arm1
COL_BASE_A2 = (255, 200, 0)   # cyan for arm2
COL_TIP_L   = (0, 255, 0)     # green
COL_TIP_R   = (255, 0, 255)   # magenta
COL_CLICK   = (0, 255, 255)   # yellow ring for user's clicks

def project_points_for_arm(s, arm):
    """Return dict of landmark->pixel for a given arm at this frame.
       arm=1 applies the arm1->arm2 transform; arm=2 is used as-is."""
    if arm == 1:
        pos = np.array(s[7:10]); q = s[10:14]; gw = float(s[15])
    else:
        pos = np.array(s[23:26]); q = s[26:30]; gw = float(s[31])
    if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
       or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
        return None
    R_g = quat_to_R(q)
    base = pos + R_g @ BASE_OFFSET
    tl, tr = tip_offsets(gw)
    tip_l = pos + R_g @ tl
    tip_r = pos + R_g @ tr
    # Bring this arm's pose into the calibrated arm's frame (if different)
    if arm != CAL_ARM:
        if CAL_ARM == 2:   # arm here is 1 → transform to arm2
            R_t, t_t = R_a1_to_a2, t_a1_to_a2
        else:              # arm here is 2 → transform to arm1
            R_t, t_t = R_a2_to_a1, t_a2_to_a1
        base   = R_t @ base   + t_t
        tip_l  = R_t @ tip_l  + t_t
        tip_r  = R_t @ tip_r  + t_t
    pts = np.stack([base, tip_l, tip_r], axis=0)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    return {"base": proj[0], "tip_l": proj[1], "tip_r": proj[2]}

t0 = time.time()
for i, fp in enumerate(frames):
    img = cv2.imread(str(fp))
    s = meta[i]["state"]
    if s is None:
        cv2.imwrite(str(OUT / f"{i:06d}.png"), img); continue

    for arm, col_base in [(1, COL_BASE_A1), (2, COL_BASE_A2)]:
        pts = project_points_for_arm(s, arm)
        if pts is None: continue
        # Connect tip-L .. base .. tip-R with a white line for "finger shape"
        pl = tuple(int(v) for v in pts["tip_l"])
        pb = tuple(int(v) for v in pts["base"])
        pr = tuple(int(v) for v in pts["tip_r"])
        cv2.line(img, pl, pb, (255,255,255), 2, cv2.LINE_AA)
        cv2.line(img, pb, pr, (255,255,255), 2, cv2.LINE_AA)
        # Tip markers (smaller)
        for p, col in [(pl, COL_TIP_L), (pr, COL_TIP_R)]:
            cv2.circle(img, p, 8, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, p, 7, col, -1, cv2.LINE_AA)
        # Base marker (PROMINENT — what the user wanted)
        cv2.circle(img, pb, 14, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(img, pb, 12, col_base, -1, cv2.LINE_AA)
        cv2.circle(img, pb, 5, (255,255,255), -1, cv2.LINE_AA)
        cv2.putText(img, f'a{arm}', (pb[0]+16, pb[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_base, 1, cv2.LINE_AA)

    # If this frame was clicked, also draw the click rings for comparison
    for c in clicks_by_frame.get(i, []):
        cu, cv_ = int(c["u"]), int(c["v"])
        cv2.circle(img, (cu, cv_), 10, COL_CLICK, 2, cv2.LINE_AA)
        cv2.putText(img, f'a{c["arm"]} click', (cu+12, cv_+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_CLICK, 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT / f"{i:06d}.png"), img)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(frames)}  {time.time()-t0:.1f}s")

print(f"done in {time.time()-t0:.1f}s, wrote {len(frames)} overlays")
