"""Render the YAM gripper outline using cv2.projectPoints — same camera math
as the cyan debug markers, so any offset from the markers comes purely from
the gripper's 3D model, not from a mujoco↔OpenCV camera-convention mismatch.

We sample the gripper.stl + tip_left.stl + tip_right.stl meshes, project each
vertex through the calibrated camera, and draw filled silhouettes.

Output: artifacts/proj_overlay/*.png  (overwrites the existing overlays so
the viewer's debug panel shows this directly)
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import trimesh
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
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
CAL_ARM = calib.get("calibrated_arm", 2)

# Use the linear_4310 Mujoco model so we can read body transforms
# (body offsets + geom offsets + mesh vertices) at any gripper width.
ASSETS = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "assets"
GRIPPER_XML = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "linear_4310.xml"
grip_m = MjModel.from_xml_path(str(GRIPPER_XML))
grip_d = MjData(grip_m)

# Load the STL vertices once
def load_mesh(name):
    m = trimesh.load(str(ASSETS / name), force='mesh')
    return np.asarray(m.vertices, dtype=np.float64)

MESH = {
    "gripper": load_mesh("gripper.stl"),
    "tip_left": load_mesh("tip_left.stl"),
    "tip_right": load_mesh("tip_right.stl"),
}

# Geom id by body name
GEOM_BY_BODY = {}
for i in range(grip_m.ngeom):
    bn = mujoco.mj_id2name(grip_m, mujoco.mjtObj.mjOBJ_BODY, grip_m.geom_bodyid[i])
    GEOM_BY_BODY.setdefault(bn, []).append(i)

def gripper_vertices_in_gripper_frame(grip_w):
    """Return a dict {body_name: Nx3 vertices in gripper-body frame} with
    the given jaw width applied (slide joints)."""
    half = float(np.clip(grip_w * 0.5, 0.0, 0.0475))
    grip_d.qpos[:] = [half, half]
    mujoco.mj_forward(grip_m, grip_d)
    out = {}
    name_to_mesh = {"gripper": "gripper", "tip_left": "tip_left", "tip_right": "tip_right"}
    for body_name, mesh_key in name_to_mesh.items():
        gi = GEOM_BY_BODY[body_name][0]
        R = grip_d.geom_xmat[gi].reshape(3, 3)  # mesh-local -> gripper-body frame
        t = grip_d.geom_xpos[gi].copy()         # geom origin in gripper-body frame
        v_local = MESH[mesh_key]
        v_grip = (R @ v_local.T).T + t          # gripper-body frame
        out[body_name] = v_grip
    return out

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

# Pre-index clicks
clicks_by_frame = {}
for c in clicks:
    clicks_by_frame.setdefault(c["frame"], []).append(c)

OUT = ROOT / "artifacts" / "proj_overlay"
OUT.mkdir(exist_ok=True, parents=True)
FRAMES = sorted((ROOT / "artifacts" / "pinhole").glob("*.png"))

# Colors (BGR)
COL_BASE_A1 = (0, 220, 255)    # arm1 base — cyan-yellow
COL_BASE_A2 = (255, 200, 0)    # arm2 base — cyan
COL_GRIPPER_A1 = (50, 180, 255)  # arm1 gripper outline — light orange
COL_GRIPPER_A2 = (255, 220, 100) # arm2 gripper outline — light blue
COL_CLICK = (0, 255, 255)
BASE_OFFSET = np.array([0.0, 0.0, -0.0546], dtype=np.float64)

def gripper_pose_in_cal_frame(s, vive_arm):
    if vive_arm == 1:
        pos = np.array(s[7:10]); q = s[10:14]; gw = float(s[15])
    else:
        pos = np.array(s[23:26]); q = s[26:30]; gw = float(s[31])
    if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
       or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
        return None
    R_g = quat_to_R(q)
    # Bring into calibrated arm's frame
    if vive_arm != CAL_ARM:
        if CAL_ARM == 1:
            Rt, tt = R_a2_to_a1, t_a2_to_a1
        else:
            Rt, tt = R_a1_to_a2, t_a1_to_a2
        pos = Rt @ pos + tt
        R_g = Rt @ R_g
    return pos, R_g, gw

def draw_gripper(img, pos, R_g, gw, fill_col, line_col):
    """Project the gripper mesh vertices and render as a hull silhouette."""
    parts = gripper_vertices_in_gripper_frame(gw)  # in gripper body frame
    all_world = []
    for v_grip in parts.values():
        all_world.append((R_g @ v_grip.T).T + pos)
    pts_world = np.concatenate(all_world, axis=0)
    pts_2d, _ = cv2.projectPoints(pts_world, rvec, tvec, K, None)
    pts_2d = pts_2d.reshape(-1, 2)
    # Filter to image bounds
    H, W = img.shape[:2]
    finite = np.isfinite(pts_2d).all(axis=1)
    pts_2d = pts_2d[finite]
    if len(pts_2d) < 5:
        return
    # Compute the convex hull for a clean silhouette
    pts_i = pts_2d.astype(np.int32)
    hull = cv2.convexHull(pts_i.reshape(-1, 1, 2))
    overlay = img.copy()
    cv2.fillPoly(overlay, [hull], fill_col)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
    cv2.polylines(img, [hull], True, line_col, 2, cv2.LINE_AA)

t0 = time.time()
for i, fp in enumerate(FRAMES):
    img = cv2.imread(str(fp))
    s = meta[i]["state"]
    if s is None:
        cv2.imwrite(str(OUT / f"{i:06d}.png"), img); continue

    for vive_arm, fill, line, col_base in [
        (1, (60, 60, 160), (50, 100, 255),  COL_BASE_A1),
        (2, (160, 100, 60), (255, 220, 100), COL_BASE_A2),
    ]:
        pose = gripper_pose_in_cal_frame(s, vive_arm)
        if pose is None: continue
        pos, R_g, gw = pose
        draw_gripper(img, pos, R_g, gw, fill_col=fill, line_col=line)
        # Bright base marker on top of the gripper silhouette
        base_world = pos + R_g @ BASE_OFFSET
        b2, _ = cv2.projectPoints(base_world.reshape(1, 3), rvec, tvec, K, None)
        bu, bv = b2[0, 0]
        if np.isfinite(bu) and np.isfinite(bv):
            cv2.circle(img, (int(bu), int(bv)), 12, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(bu), int(bv)), 10, col_base, -1, cv2.LINE_AA)
            cv2.putText(img, f"a{vive_arm}", (int(bu)+14, int(bv)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_base, 1, cv2.LINE_AA)

    # Click rings (for clicked frames)
    for c in clicks_by_frame.get(i, []):
        cu, cv_ = int(c["u"]), int(c["v"])
        cv2.circle(img, (cu, cv_), 10, COL_CLICK, 2, cv2.LINE_AA)
        cv2.putText(img, f'a{c["arm"]} click', (cu+12, cv_+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_CLICK, 1, cv2.LINE_AA)

    cv2.imwrite(str(OUT / f"{i:06d}.png"), img)
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(FRAMES)}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s, wrote {len(FRAMES)} overlays")
