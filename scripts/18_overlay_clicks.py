"""Overlay user clicks (yellow) and the calibrated projection of each
landmark (cyan) on top of the debug composite. If clicks and projections
match → calibration fits the clicks. If they don't → PnP failed at those
clicks (landmark offset mismatch or RANSAC outlier).
"""
import json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

ROOT = Path(__file__).resolve().parent.parent
calib = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
K = np.array(calib["K"]); rvec = np.array(calib["rvec"]); tvec = np.array(calib["tvec"])
R_a1_to_a2 = np.array(calib.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(calib.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
meta = json.loads((ROOT / "data" / "frames_meta.json").read_text())
clicks = json.loads((ROOT / "data" / "calibration_clicks.json").read_text())["clicks"]

# Gripper geometry (same as solver)
GRIPPER_XML = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "linear_4310.xml"
grip_m = MjModel.from_xml_path(str(GRIPPER_XML))
grip_d = MjData(grip_m)
GEOM_ID = {
    "tip_l": next(i for i in range(grip_m.ngeom)
                  if mujoco.mj_id2name(grip_m, mujoco.mjtObj.mjOBJ_BODY,
                                        grip_m.geom_bodyid[i]) == "tip_left"),
    "tip_r": next(i for i in range(grip_m.ngeom)
                  if mujoco.mj_id2name(grip_m, mujoco.mjtObj.mjOBJ_BODY,
                                        grip_m.geom_bodyid[i]) == "tip_right"),
}
def landmark_offset(name, grip_w):
    if name == "base":
        return np.array([0, 0, -0.0546], dtype=np.float64)
    if name in ("tip_l", "tip_r"):
        half = float(np.clip(grip_w*0.5, 0, 0.0475))
        grip_d.qpos[:] = [half, half]
        mujoco.mj_forward(grip_m, grip_d)
        return np.asarray(grip_d.geom_xpos[GEOM_ID[name]], dtype=np.float64).copy()
    return np.zeros(3)
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

IN = ROOT / "artifacts" / "debug_composite"
OUT = ROOT / "artifacts" / "debug_overlay"
OUT.mkdir(exist_ok=True, parents=True)

# Group clicks by frame
by_frame = {}
for c in clicks:
    by_frame.setdefault(c["frame"], []).append(c)

# For each frame we have clicks on, write an overlay with click vs projection
for fr, cs in sorted(by_frame.items()):
    src = IN / f"{fr:06d}.png"
    if not src.exists(): continue
    img = cv2.imread(str(src))
    s = meta[fr]["state"]
    if s is None: continue
    for c in cs:
        arm = c["arm"]; land = c["land"]
        if arm == 1:
            pos = np.array(s[7:10]); q = s[10:14]; gw = float(s[15])
            R_g = quat_to_R(q)
            Pw_native = pos + R_g @ landmark_offset(land, gw)
            # Transform arm1's native frame into the calibrated (arm2) frame
            Pw = R_a1_to_a2 @ Pw_native + t_a1_to_a2
        else:
            pos = np.array(s[23:26]); q = s[26:30]; gw = float(s[31])
            Pw = pos + quat_to_R(q) @ landmark_offset(land, gw)
        proj, _ = cv2.projectPoints(Pw.reshape(1,3), rvec, tvec, K, None)
        pu, pv = proj[0,0]
        cu, cv_ = c["u"], c["v"]
        # YELLOW click
        cv2.circle(img, (int(cu), int(cv_)), 6, (0, 255, 255), 2)
        cv2.putText(img, f'{land}', (int(cu)+8, int(cv_)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
        # CYAN projection
        cv2.circle(img, (int(pu), int(pv)), 5, (255, 200, 0), -1)
        cv2.line(img, (int(cu), int(cv_)), (int(pu), int(pv)), (255,255,255), 1)
        err = ((pu-cu)**2 + (pv-cv_)**2)**0.5
        print(f"f{fr} a{arm} {land:6s} click=({cu:.0f},{cv_:.0f}) proj=({pu:.0f},{pv:.0f}) err={err:.0f}px")
    cv2.imwrite(str(OUT / f"{fr:06d}.png"), img)
print(f"\nwrote {len(by_frame)} overlay frames to {OUT}")
