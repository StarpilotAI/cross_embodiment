"""PnP solver: clicks are already in pinhole coords, no fisheye undistort.

Landmarks:
  tip_l: visible left fingertip   (geom centroid of tip_left mesh)
  tip_r: visible right fingertip  (geom centroid of tip_right mesh)
  base : midpoint between the two tips at the base of the fingers
         (where they emerge from the slide mechanism)
"""
import json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

ROOT = Path(__file__).resolve().parent.parent

W = H = 1024
FOV_PIN_DEG = 60.0
F_PIN = (W/2) / np.tan(np.deg2rad(FOV_PIN_DEG/2))
K_PIN = np.array([[F_PIN, 0, W/2], [0, F_PIN, H/2], [0, 0, 1]], dtype=np.float64)

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
    """Position of named landmark in the gripper's local frame."""
    if name == "base":
        # midpoint between tip BODY origins (where the fingers attach to the
        # slide mechanism). This z is constant regardless of jaw width.
        return np.array([0.0, 0.0, -0.0546], dtype=np.float64)
    if name in ("tip_l", "tip_r"):
        half = float(np.clip(grip_w * 0.5, 0.0, 0.0475))
        _grip_d.qpos[:] = [half, half]
        mujoco.mj_forward(_grip_m, _grip_d)
        return np.asarray(_grip_d.geom_xpos[_GEOM_ID[name]], dtype=np.float64).copy()
    # legacy
    if name == "origin":
        return np.zeros(3, dtype=np.float64)
    raise ValueError(f"unknown landmark {name}")

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
    print(f"loaded {len(clicks)} clicks (pinhole space)")

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
    for c, why in skipped[:10]:
        print(f"  skip f{c['frame']} a{c['arm']} {c['land']}: {why}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped)-10} more")

    obj = np.array(obj, dtype=np.float64)
    img = np.array(img, dtype=np.float64)

    if len(obj) < 4:
        raise SystemExit("need >=4 correspondences for PnP")

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K_PIN, distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=20.0, iterationsCount=500,
    )
    if not ok:
        raise SystemExit("PnP RANSAC failed")
    print(f"RANSAC inliers: {len(inliers)}/{len(obj)}")

    obj_in = obj[inliers[:,0]]; img_in = img[inliers[:,0]]
    ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K_PIN, None, rvec, tvec,
                                   useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K_PIN, None)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - img_in)**2, axis=1)))
    print(f"RMS reproj err: {err:.2f} px")

    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec.flatten()
    print(f"cam pos world: {cam_pos}")
    print(f"cam view dir (world):  {-R.T @ np.array([0,0,1])}")

    out = {
        "K": K_PIN.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "R_world_to_cam": R.tolist(),
        "cam_pos_world": cam_pos.tolist(),
        "reproj_rms_px": float(err),
        "n_inliers": int(len(inliers)),
        "n_correspondences": int(len(obj)),
        "image_size": [W, H],
        "fov_pinhole_deg": FOV_PIN_DEG,
        "note": "clicks were on pinhole-undistorted frames (artifacts/pinhole/*)",
    }
    out_path = ROOT / "data" / "scene_cam_calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
