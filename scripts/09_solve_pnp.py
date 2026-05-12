"""Solve scene-cam pose from calibrate.html clicks via cv2.solvePnP.

Input  : data/calibration_clicks.json  (downloaded from the viewer)
Vive 3D: data/frames_meta.json  (per-frame arm{1,2}_pose_xyz/quat)

For each click record (frame, arm, landmark, u, v):
  P_world = T_vive(frame, arm) @ P_landmark_offset
where P_landmark_offset is the position of the named landmark in the gripper's
local frame. The Vive pose (pos + quat) gives the gripper's pose in world.

We then run cv2.solvePnP (or solvePnPRansac) on the (P_world, (u,v)) pairs to
recover the scene camera. If --refine_intrinsics is set, we use solvePnP with a
free focal length; otherwise we assume a pinhole with the FOV given on the cmdline.

Outputs (data/scene_cam_calibration.json):
  K       : 3x3 intrinsics
  rvec    : 3x1 rotation vector (world->cam)
  tvec    : 3x1 translation (world->cam)
  reproj  : RMS pixel reprojection error
"""
import json, argparse
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

ROOT = Path(__file__).resolve().parent.parent

# The PIKA gripper in the dataset has the SAME geometry as the YAM
# linear_4310 gripper, so we read landmark positions directly from that
# Mujoco model rather than guessing offsets.
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

def landmark_offset(name: str, gripper_distance_m: float) -> np.ndarray:
    """Return the (x,y,z) of a landmark in the gripper's local frame.
    `gripper_distance_m` is the dataset's recorded jaw separation (~0..0.095)."""
    if name == "origin":
        return np.zeros(3, dtype=np.float64)
    half = float(np.clip(gripper_distance_m * 0.5, 0.0, 0.0475))
    _grip_d.qpos[:] = [half, half]
    mujoco.mj_forward(_grip_m, _grip_d)
    return np.asarray(_grip_d.geom_xpos[_GEOM_ID[name]], dtype=np.float64).copy()

def quat_to_R(q):
    """quat (qx,qy,qz,qw) → 3x3 rotation"""
    qx, qy, qz, qw = q
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", default=str(ROOT / "data" / "calibration_clicks.json"))
    ap.add_argument("--meta",   default=str(ROOT / "data" / "frames_meta.json"))
    ap.add_argument("--fov_init", type=float, default=140.0,
                    help="Initial guess of scene-cam horizontal FOV in degrees")
    ap.add_argument("--refine_intrinsics", action="store_true",
                    help="Also refine focal length (needs more clicks)")
    ap.add_argument("--out", default=str(ROOT / "data" / "scene_cam_calibration.json"))
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    data = json.loads(Path(args.clicks).read_text())
    clicks = data["clicks"]
    W, H = data["image_size"]
    print(f"loaded {len(clicks)} clicks on {W}x{H} images")

    # Build (P_world, (u,v)) correspondences
    obj = []   # 3D world points
    img = []   # 2D image points
    skipped = 0
    for c in clicks:
        f = c["frame"]; arm = c["arm"]; land = c["land"]
        u, v = float(c["u"]), float(c["v"])
        m = meta[f]
        if m["state"] is None:
            skipped += 1; continue
        s = m["state"]
        if arm == 1:
            pos = np.array(s[7:10])
            qx, qy, qz, qw = s[10], s[11], s[12], s[13]
            grip_w = float(s[15])
        else:
            pos = np.array(s[7+16:10+16])
            qx, qy, qz, qw = s[10+16], s[11+16], s[12+16], s[13+16]
            grip_w = float(s[15+16])
        # Skip frames with degenerate quaternion (sometimes happens at frame 0)
        if abs(qw - 1.0) < 1e-6 and abs(qx) < 1e-6 and abs(qy) < 1e-6 and abs(qz) < 1e-6:
            skipped += 1; continue
        if np.linalg.norm(pos) < 1e-6:
            skipped += 1; continue
        # Outlier guard for arm2 (we saw values up to 10m in this dataset)
        if np.max(np.abs(pos)) > 1.5:
            skipped += 1; continue
        R_gripper = quat_to_R([qx, qy, qz, qw])
        Pw = pos + R_gripper @ landmark_offset(land, grip_w)
        obj.append(Pw)
        img.append([u, v])
    obj = np.array(obj, dtype=np.float64)
    img = np.array(img, dtype=np.float64)
    print(f"usable correspondences: {len(obj)} (skipped {skipped})")
    if len(obj) < 4:
        raise SystemExit("need at least 4 valid correspondences for PnP")

    # Initial intrinsics from FOV guess
    f_pix = (W/2) / np.tan(np.deg2rad(args.fov_init/2))
    K = np.array([[f_pix, 0, W/2],
                  [0, f_pix, H/2],
                  [0,     0,   1]], dtype=np.float64)
    print(f"initial K: f={f_pix:.1f} px (FOV={args.fov_init}deg)")

    # solvePnPRansac is robust to a few bad clicks
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=20.0, iterationsCount=300,
    )
    if not ok:
        raise SystemExit("solvePnPRansac failed")
    print(f"ransac OK · inliers {len(inliers)}/{len(obj)}")

    # Refine using only inliers
    obj_in = obj[inliers[:,0]]
    img_in = img[inliers[:,0]]
    ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K, None,
                                   rvec, tvec, useExtrinsicGuess=True,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K, None)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - img_in)**2, axis=1)))
    print(f"after refine: RMS reproj err = {err:.2f} px")

    # Decompose
    R, _ = cv2.Rodrigues(rvec)
    cam_pos_world = -R.T @ tvec.flatten()
    print(f"camera position in world: {cam_pos_world}")
    print(f"R (world->cam):\n{R}")

    out = {
        "K": K.tolist(),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "R_world_to_cam": R.tolist(),
        "cam_pos_world": cam_pos_world.tolist(),
        "reproj_rms_px": float(err),
        "n_inliers": int(len(inliers)),
        "n_correspondences": int(len(obj)),
        "image_size": [W, H],
        "fov_init_deg": args.fov_init,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
