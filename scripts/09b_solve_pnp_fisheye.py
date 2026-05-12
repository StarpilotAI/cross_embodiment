"""PnP solve with fisheye-aware click undistortion.

The clicks were collected on the raw fisheye scene_rgb_2 image (heavy radial
distortion). We undistort each click (u,v) from fisheye to pinhole coords
using the same fisheye model used in scripts/07_undistort.py, then solve a
normal pinhole PnP.

The output camera is a PINHOLE camera in the same projection as the pinhole
frames in artifacts/pinhole/, so the Mujoco render will align with those.
"""
import json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

ROOT = Path(__file__).resolve().parent.parent

# --- Fisheye model (matches scripts/07_undistort.py) ----
W = H = 1024
F_FISH = 325.0
K_FISH = np.array([[F_FISH, 0, W/2], [0, F_FISH, H/2], [0, 0, 1]], dtype=np.float64)
D_FISH = np.zeros(4, dtype=np.float64)
# --- Pinhole target (60deg FOV) ----
FOV_PIN_DEG = 60.0
F_PIN = (W/2) / np.tan(np.deg2rad(FOV_PIN_DEG/2))
K_PIN = np.array([[F_PIN, 0, W/2], [0, F_PIN, H/2], [0, 0, 1]], dtype=np.float64)

# --- Gripper geometry (same as 09_solve_pnp.py) ----
GRIPPER_XML = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "linear_4310.xml"
_grip_m = MjModel.from_xml_path(str(GRIPPER_XML))
_grip_d = MjData(_grip_m)
_GEOM_ID = {
    "tip_l": next(i for i in range(_grip_m.ngeom)
                  if mujoco.mj_id2name(_grip_m, mujoco.mjtObj.mjOBJ_BODY,
                                        _grip_m.geom_bodyid[i]) == "tip_left"),
    "tip_r": next(i for i in range(_grip_m.ngeom)
                  if mujoco.mj_id2name(_grip_m, mujoco.mjtObj.mjOBJ_BODY,
                                        _grip_m.geom_bodyid[i]) == "tip_right"),
}

def landmark_offset(name, grip_w):
    if name == "origin":
        return np.zeros(3, dtype=np.float64)
    half = float(np.clip(grip_w * 0.5, 0.0, 0.0475))
    _grip_d.qpos[:] = [half, half]
    mujoco.mj_forward(_grip_m, _grip_d)
    return np.asarray(_grip_d.geom_xpos[_GEOM_ID[name]], dtype=np.float64).copy()

def quat_to_R(q):
    qx, qy, qz, qw = q
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])

def main():
    meta   = json.loads((ROOT / "data" / "frames_meta.json").read_text())
    clicks = json.loads((ROOT / "data" / "calibration_clicks.json").read_text())["clicks"]
    print(f"loaded {len(clicks)} clicks")

    obj, img = [], []
    skipped = []
    for c in clicks:
        f = c["frame"]; arm = c["arm"]; land = c["land"]
        u, v = float(c["u"]), float(c["v"])
        m = meta[f]
        if m["state"] is None:
            skipped.append((c, "no state")); continue
        s = m["state"]
        if arm == 1:
            pos = np.array(s[7:10]); q = s[10:14]; gw = float(s[15])
        else:
            pos = np.array(s[7+16:10+16]); q = s[10+16:14+16]; gw = float(s[15+16])
        if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
           or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
            skipped.append((c, "bad vive pose")); continue
        Pw = pos + quat_to_R(q) @ landmark_offset(land, gw)
        obj.append(Pw)
        img.append([u, v])
    print(f"usable: {len(obj)}, skipped: {len(skipped)}")
    for c, why in skipped:
        print(f"  skip f{c['frame']} a{c['arm']} {c['land']}: {why}")

    obj = np.array(obj, dtype=np.float64)
    img_fish = np.array(img, dtype=np.float64).reshape(-1, 1, 2)

    # Undistort the click pixels from fisheye coords to pinhole coords.
    # cv2.fisheye.undistortPoints expects shape (N,1,2). Pass Knew=K_PIN so
    # output is in pinhole pixel space (not normalised).
    img_pin = cv2.fisheye.undistortPoints(img_fish, K_FISH, D_FISH, P=K_PIN).reshape(-1, 2)
    print(f"undistorted clicks (first 3):")
    for i in range(min(3, len(img_pin))):
        print(f"  fish({img_fish[i,0,0]:.0f},{img_fish[i,0,1]:.0f}) -> pin({img_pin[i,0]:.0f},{img_pin[i,1]:.0f})")

    # PnP in pinhole space
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img_pin, K_PIN, distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=20.0, iterationsCount=500,
    )
    if not ok:
        raise SystemExit("PnP RANSAC failed")
    print(f"RANSAC inliers: {len(inliers)}/{len(obj)}")

    obj_in = obj[inliers[:,0]]; img_in = img_pin[inliers[:,0]]
    ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K_PIN, None, rvec, tvec,
                                   useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K_PIN, None)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - img_in)**2, axis=1)))
    print(f"RMS reproj err (pinhole space): {err:.2f} px")

    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec.flatten()
    print(f"cam pos (world): {cam_pos}")
    print(f"forward dir (world): {-R.T @ np.array([0,0,1])}")

    out = {
        "K": K_PIN.tolist(),
        "K_fisheye": K_FISH.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "R_world_to_cam": R.tolist(),
        "cam_pos_world": cam_pos.tolist(),
        "reproj_rms_px": float(err),
        "n_inliers": int(len(inliers)),
        "n_correspondences": int(len(obj)),
        "image_size": [W, H],
        "fov_pinhole_deg": FOV_PIN_DEG,
        "note": "pinhole calibration aligned with artifacts/pinhole/* frames",
    }
    out_path = ROOT / "data" / "scene_cam_calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
