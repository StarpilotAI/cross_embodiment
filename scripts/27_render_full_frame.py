"""Single-frame full-bimanual YAM render + IK, for fast debugging.

Reuses the base-placement convention from scripts/26_three_options.py
(option B: world_up = cross(side_axis, R_g1[:,2]) computed from frame 94,
where the two PIKA grippers were resting on the floor side-by-side). Bases
are placed behind the workspace and below it on that floor plane.

Usage:
  .venv/Scripts/python.exe scripts/27_render_full_frame.py 376
  .venv/Scripts/python.exe scripts/27_render_full_frame.py 376 0.25 0.10
                                                             back  down
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 376
BACK  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
DOWN  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
BASE_YAW_DEG = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
CAL_FRAME = 94   # frame where grippers rest on the floor side-by-side

CALIB = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
META  = json.loads((ROOT / "data" / "frames_meta.json").read_text())

K = np.array(CALIB["K"], dtype=np.float64)
rvec = np.array(CALIB["rvec"], dtype=np.float64)
tvec = np.array(CALIB["tvec"], dtype=np.float64)
W, H = CALIB["image_size"]
R_a1_to_a2 = np.array(CALIB["R_arm1_to_arm2"], dtype=np.float64)
t_a1_to_a2 = np.array(CALIB["t_arm1_to_arm2"], dtype=np.float64)
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
CAL_ARM = CALIB.get("calibrated_arm", 2)
fovy_deg = float(np.degrees(2 * np.arctan2(H/2, K[1,1])))

R_w2c, _ = cv2.Rodrigues(rvec)
R_c2w = R_w2c.T
cam_pos = -R_c2w @ tvec
mj_x = R_c2w[:, 0]
mj_y = -R_c2w[:, 1]
xyaxes_cam = " ".join(f"{v:.10f}" for v in list(mj_x) + list(mj_y))
pos_cam_str = " ".join(f"{v:.10f}" for v in cam_pos)


def quat_R(q):
    qx, qy, qz, qw = q
    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def gripper_in_cal(s, vive_arm):
    """Return (pos, R) of vive_arm's gripper in the calibration frame.
    Returns None if pose looks invalid (identity / zero / out-of-range)."""
    if vive_arm == 1:
        pos = np.array(s[7:10]); q = s[10:14]
    else:
        pos = np.array(s[23:26]); q = s[26:30]
    if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
       or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
        return None
    R = quat_R(q)
    if vive_arm != CAL_ARM:
        if CAL_ARM == 1:
            Rt, tt = R_a2_to_a1, t_a2_to_a1
        else:
            Rt, tt = R_a1_to_a2, t_a1_to_a2
        pos = Rt @ pos + tt
        R = Rt @ R
    return pos, R


# --- Step 1: derive floor frame from CAL_FRAME (94) --------------------------
s_cal = META[CAL_FRAME]["state"]
p1_cal, R1_cal = gripper_in_cal(s_cal, 1)
p2_cal, R2_cal = gripper_in_cal(s_cal, 2)
side_axis = p2_cal - p1_cal
side_axis /= np.linalg.norm(side_axis)
fwd_guess = R1_cal[:, 2]   # gripper's local +z direction in cal frame
WORLD_UP = np.cross(side_axis, fwd_guess)
WORLD_UP /= np.linalg.norm(WORLD_UP)

# Forward in the floor plane (away from grippers, behind them)
fwd = fwd_guess - np.dot(fwd_guess, WORLD_UP) * WORLD_UP
fwd /= np.linalg.norm(fwd)
BEHIND_DIR = -fwd
DOWN_DIR = -WORLD_UP

# Floor-aligned base axes (consistent across both bases), then yaw around UP
base_x0 = side_axis - np.dot(side_axis, WORLD_UP) * WORLD_UP
base_x0 /= np.linalg.norm(base_x0)
base_y0 = np.cross(WORLD_UP, base_x0)
base_y0 /= np.linalg.norm(base_y0)
_th = np.radians(BASE_YAW_DEG)
base_x = np.cos(_th) * base_x0 + np.sin(_th) * base_y0
base_y = -np.sin(_th) * base_x0 + np.cos(_th) * base_y0
base_xyaxes = " ".join(f"{v:.10f}" for v in list(base_x) + list(base_y))
print(f"BASE_YAW = {BASE_YAW_DEG}deg")
print(f"base_x = {base_x}")
print(f"base_y = {base_y}")

# --- Step 2: place bases relative to each frame-94 gripper position ----------
B1 = p1_cal + BACK * BEHIND_DIR + DOWN * DOWN_DIR
B2 = p2_cal + BACK * BEHIND_DIR + DOWN * DOWN_DIR
print(f"calib frame = {CAL_FRAME}")
print(f"  WORLD_UP   = {WORLD_UP}")
print(f"  BEHIND_DIR = {BEHIND_DIR}")
print(f"  g1@94 = {p1_cal}  g2@94 = {p2_cal}")
print(f"  back={BACK}  down={DOWN}")
print(f"  B1 = {B1}")
print(f"  B2 = {B2}")

# --- Step 3: target poses at the chosen render frame -------------------------
# Canonical i2rt mounting puts gripper body at link6 origin (pos=0,quat=I).
# To match script 21's mocap render (where the gripper geometry sits at
# (P_vive, R_vive) with an internal Rx(180°) flip), the IK target for the
# link6 site must be
#   P_link6  = P_vive + R_vive @ (0, 0, -0.1092)
#   R_link6  = R_vive @ Rx(180°)
R_GFIX = np.diag([1.0, -1.0, -1.0])
GFIX_OFFSET = np.array([0.0, 0.0, -0.1092])
s_render = META[FRAME]["state"]
assert s_render is not None, f"frame {FRAME} has no state"
tgt1 = gripper_in_cal(s_render, 1)
tgt2 = gripper_in_cal(s_render, 2)
assert tgt1 is not None and tgt2 is not None, f"missing pose at frame {FRAME}"
p1_vive, R1_vive = tgt1
p2_vive, R2_vive = tgt2
p1 = p1_vive + R1_vive @ GFIX_OFFSET
p2 = p2_vive + R2_vive @ GFIX_OFFSET
R1 = R1_vive @ R_GFIX
R2 = R2_vive @ R_GFIX
print(f"\nrender frame = {FRAME}")
print(f"  gripper1 Vive pos = {p1_vive}  -> link6 target {p1}")
print(f"  gripper2 Vive pos = {p2_vive}  -> link6 target {p2}")

# --- Step 4: build the bimanual scene XML ------------------------------------
YAM_ASSETS = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam" / "assets"


def yam_chain(prefix, base_pos, xyaxes):
    bx, by, bz = base_pos
    return f"""
    <body name="{prefix}_base" pos="{bx} {by} {bz}" xyaxes="{xyaxes}">
      <geom class="yam_dark" pos="-0.0374966 -0.0464005 0.187501" quat="0.707105 0 0.707108 0" mesh="base"/>
      <body name="{prefix}_link1" pos="0 0 0.067" quat="0.707105 0 0 -0.707108">
        <joint name="{prefix}_j1" pos="0 0 0" axis="0 0 1" range="-2.61799 3.05433"/>
        <geom class="yam_link" pos="-0.0464 0.0374968 0.119501" quat="0.499998 0.5 0.5 -0.500002" mesh="link1"/>
        <body name="{prefix}_link2" pos="-0.0329 0.02 0.0455" quat="0.499998 0.5 -0.500002 -0.5">
          <joint name="{prefix}_j2" pos="0 0 0" axis="0 0 1" range="0 3.65"/>
          <geom class="yam_link" pos="-0.0174977 -0.0740001 -0.07925" quat="0.499998 0.5 0.500002 0.5" mesh="link2"/>
          <body name="{prefix}_link3" pos="0.264 4.08431e-07 -0.06375" quat="9.38184e-07 -0.707105 -0.707108 9.38187e-07">
            <joint name="{prefix}_j3" pos="0 0 0" axis="0 0 1" range="0 3.66519"/>
            <geom class="yam_link" pos="0.0740003 -0.281499 -0.0813" quat="9.38184e-07 9.38187e-07 -0.707108 -0.707105" mesh="link3"/>
            <body name="{prefix}_link4" pos="0.0600003 -0.244999 -0.00205" quat="1.32679e-06 0 0 -1">
              <joint name="{prefix}_j4" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
              <geom class="yam_link" pos="-0.0138003 0.0364989 -0.0787882" quat="0.707105 0.707108 0 0" mesh="link4"/>
              <body name="{prefix}_link5" pos="-0.0403003 0.0703851 -0.0323887" quat="9.38184e-07 -0.707105 -9.38187e-07 -0.707108">
                <joint name="{prefix}_j5" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
                <geom class="yam_dark" pos="-0.0463995 0.0311519 0.0265" quat="0.499998 -0.5 -0.5 -0.500002" mesh="link5"/>
                <body name="{prefix}_link6" pos="2.39858e-07 -0.0419481 0.0404996" quat="0.499998 -0.5 -0.5 -0.500002">
                  <joint name="{prefix}_j6" pos="0 0 0" axis="0 0 -1" range="-2.0944 2.0944"/>
                  <site name="{prefix}_ee" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
                  <body name="{prefix}_gripper" pos="0 0 0" quat="1 0 0 0">
                    <geom pos="-0.014 -0.0463995 0.0731" quat="1 0 0 0" type="mesh" mesh="gripper_body" material="dark" contype="0" conaffinity="0"/>
                    <body name="{prefix}_tl" pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002">
                      <geom pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="tip_left" material="alum" contype="0" conaffinity="0"/>
                    </body>
                    <body name="{prefix}_tr" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">
                      <geom pos="-0.0379932 0.129783 0.00133753" quat="0.707105 -0.707108 0 0" type="mesh" mesh="tip_right" material="alum" contype="0" conaffinity="0"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
"""


SCENE_XML = f"""
<mujoco model="yam_full_frame{FRAME}">
  <compiler angle="radian" meshdir="{YAM_ASSETS.as_posix()}"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="{fovy_deg:.6f}"/>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.45 0.45 0.45" specular="0.1 0.1 0.1"/>
  </visual>
  <default>
    <default class="yam_link"><geom type="mesh" rgba="0.78 0.78 0.80 1.0" material="alum"/></default>
    <default class="yam_dark"><geom type="mesh" rgba="0.20 0.20 0.22 1.0" material="dark"/></default>
  </default>
  <asset>
    <mesh name="base"  file="base.stl"/>
    <mesh name="link1" file="link1.stl"/>
    <mesh name="link2" file="link2.stl"/>
    <mesh name="link3" file="link3.stl"/>
    <mesh name="link4" file="link4.stl"/>
    <mesh name="link5" file="link5.stl"/>
    <mesh name="gripper_body" file="gripper.stl"/>
    <mesh name="tip_left"  file="tip_left.stl"/>
    <mesh name="tip_right" file="tip_right.stl"/>
    <material name="alum" rgba="0.78 0.78 0.80 1.0" specular="0.4" shininess="0.6"/>
    <material name="dark" rgba="0.20 0.20 0.22 1.0" specular="0.1" shininess="0.2"/>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="32" height="32"/>
  </asset>
  <worldbody>
    <camera name="scene" pos="{pos_cam_str}" xyaxes="{xyaxes_cam}" fovy="{fovy_deg:.6f}"/>
    {yam_chain("y1", B1, base_xyaxes)}
    {yam_chain("y2", B2, base_xyaxes)}
  </worldbody>
</mujoco>
"""

xml_path = ROOT / "data" / f"yam_full_f{FRAME}.xml"
xml_path.write_text(SCENE_XML)
model = MjModel.from_xml_path(str(xml_path))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)

y1_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1, 7)]
y2_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y2_j{j}") for j in range(1, 7)]
y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt]
y2_qadr = [model.jnt_qposadr[j] for j in y2_jnt]
y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt]
y2_dofadr = [model.jnt_dofadr[j] for j in y2_jnt]
y1_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
y2_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y2_ee")

ready = np.array([0.0, 1.0, 1.2, 0.0, -0.3, 0.0])
for i, a in enumerate(y1_qadr): data.qpos[a] = ready[i]
for i, a in enumerate(y2_qadr): data.qpos[a] = ready[i]


def ik_full(qadr, dofadr, jnt_ids, site, target_pos, target_R,
            n_iter_pos=200, n_iter_full=500, rot_weight=0.3):
    """Two-stage damped LS IK.  Stage A: position only.  Stage B: balanced
    6DoF where position and orientation share the residual budget.
    rot_weight scales the rotation-axis-angle error inside the 6-vector."""
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    for _ in range(n_iter_pos):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = Jp[:, dofadr]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        step = dq
        max_s = np.max(np.abs(step))
        if max_s > 0.2: step *= (0.2 / max_s)
        for i, a in enumerate(qadr): data.qpos[a] += step[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)
    for _ in range(n_iter_full):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site].copy()
        cur_R = data.site_xmat[site].reshape(3, 3).copy()
        e_pos = target_pos - cur_pos
        R_err = target_R @ cur_R.T
        tr = np.trace(R_err)
        cos_a = np.clip((tr - 1) / 2, -1, 1)
        ang = np.arccos(cos_a)
        if ang > 1e-6:
            sin_a = np.sin(ang)
            axis = np.array([R_err[2,1]-R_err[1,2],
                             R_err[0,2]-R_err[2,0],
                             R_err[1,0]-R_err[0,1]]) / (2 * sin_a)
            e_rot = rot_weight * axis * ang
        else:
            e_rot = np.zeros(3)
        err = np.concatenate([e_pos, e_rot])
        if np.linalg.norm(err) < 1e-5:
            break
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = np.vstack([Jp[:, dofadr], rot_weight * Jr[:, dofadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        step = dq
        max_s = np.max(np.abs(step))
        if max_s > 0.1: step *= (0.1 / max_s)
        for i, a in enumerate(qadr): data.qpos[a] += step[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)


def rot_error_deg(R_cur, R_tgt):
    Re = R_tgt @ R_cur.T
    tr = np.clip((np.trace(Re) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def ik_multi(arm_id, qadr, dofadr, jnt_ids, site, target_pos, target_R):
    """Try many seed configurations, return best (q, pos_err_mm, rot_err_deg)."""
    rng = np.random.default_rng(42)
    seeds = [
        np.array([ 0.0, 1.2, 1.2,  0.0, -0.3,  0.0]),
        np.array([ 0.0, 0.6, 0.6,  0.0,  0.3,  0.0]),
        np.array([ 0.5, 1.5, 0.5,  0.5, -0.5,  0.5]),
        np.array([-0.5, 1.0, 1.5, -0.5,  0.5, -0.5]),
        np.array([ 1.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
        np.array([-1.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
        np.array([ 2.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
        np.array([-2.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
        np.array([ 0.0, 2.0, 2.0,  1.0,  1.0,  1.0]),
        np.array([ 0.0, 2.5, 0.5, -1.0, -1.0,  1.0]),
    ]
    # Add random seeds
    for _ in range(10):
        s = np.zeros(6)
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            s[i] = rng.uniform(lo, hi)
        seeds.append(s)

    best = None
    for seed in seeds:
        for i, a in enumerate(qadr): data.qpos[a] = seed[i]
        ik_full(qadr, dofadr, jnt_ids, site, target_pos, target_R)
        mujoco.mj_forward(model, data)
        pos_err = np.linalg.norm(data.site_xpos[site] - target_pos) * 1000
        rot_err = rot_error_deg(data.site_xmat[site].reshape(3,3), target_R)
        score = pos_err + 5 * rot_err   # weight rot 5x: 1deg = 5mm cost
        if best is None or score < best[0]:
            q_now = np.array([data.qpos[a] for a in qadr])
            best = (score, q_now, pos_err, rot_err, seed.copy())
    return best


print("\nrunning multi-seed IK...")
best1 = ik_multi(1, y1_qadr, y1_dofadr, y1_jnt, y1_ee, p1, R1)
for i, a in enumerate(y1_qadr): data.qpos[a] = best1[1][i]
best2 = ik_multi(2, y2_qadr, y2_dofadr, y2_jnt, y2_ee, p2, R2)
for i, a in enumerate(y2_qadr): data.qpos[a] = best2[1][i]
mujoco.mj_forward(model, data)
print(f"\nbest IK: arm1 pos={best1[2]:.1f}mm rot={best1[3]:.1f}deg (seed {best1[4]})")
print(f"         arm2 pos={best2[2]:.1f}mm rot={best2[3]:.1f}deg (seed {best2[4]})")
print(f"arm1 q (rad): {np.array2string(best1[1], precision=3)}")
print(f"arm2 q (rad): {np.array2string(best2[1], precision=3)}")
err1, rerr1 = best1[2], best1[3]
err2, rerr2 = best2[2], best2[3]

renderer.update_scene(data, camera="scene")
rgb = renderer.render()
out_raw = ROOT / "data" / "inspect" / f"full_f{FRAME}_back{BACK:.2f}_down{DOWN:.2f}_raw.png"
cv2.imwrite(str(out_raw), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

bg = cv2.imread(str(ROOT / "artifacts" / "pinhole_inpainted" / f"{FRAME:06d}.png"))
fg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
mask = (luma > 8).astype(np.uint8) * 255
alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
a = np.stack([alpha]*3, axis=-1)
comp = (a * fg + (1-a) * bg).astype(np.uint8)
# --- Draw coordinate axes + infinite world axes via cv2 projection -----------
def project_pts(pts_world):
    pts = np.asarray(pts_world, np.float64).reshape(-1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
    return proj.reshape(-1, 2)

def draw_axes(img, origin, R_frame, length, thickness, labels=None):
    o = np.asarray(origin, np.float64).reshape(1, 3)
    pts = np.vstack([
        o,
        o + length * R_frame[:, 0],
        o + length * R_frame[:, 1],
        o + length * R_frame[:, 2],
    ])
    p = project_pts(pts).astype(int)
    o2 = tuple(p[0])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    names  = labels or ["x", "y", "z"]
    for i in range(3):
        cv2.arrowedLine(img, o2, tuple(p[1+i]), colors[i], thickness, tipLength=0.18)
        cv2.putText(img, names[i], tuple(p[1+i] + np.array([4, -4])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)

def draw_infinite_axis(img, direction, color, thickness=1, label=None):
    """Draw the world line through the origin along `direction`, from -100m
    to +100m, clipped to the camera's front half-space so cv2 projection is
    well-defined."""
    d = np.asarray(direction, np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    p_neg = (-100.0) * d
    p_pos = ( 100.0) * d
    # transform to camera frame (OpenCV: +z forward)
    a_cam = R_w2c @ p_neg + tvec.reshape(3)
    b_cam = R_w2c @ p_pos + tvec.reshape(3)
    # clip segment to z >= eps
    eps = 0.05
    az, bz = a_cam[2], b_cam[2]
    if az < eps and bz < eps:
        return  # entirely behind the camera
    if az < eps:
        t = (eps - az) / (bz - az)
        a_cam = a_cam + t * (b_cam - a_cam)
    elif bz < eps:
        t = (eps - bz) / (az - bz)
        b_cam = b_cam + t * (a_cam - b_cam)
    def project_cam(pc):
        x = (K[0, 0] * pc[0] + K[0, 2] * pc[2]) / pc[2]
        y = (K[1, 1] * pc[1] + K[1, 2] * pc[2]) / pc[2]
        return int(round(x)), int(round(y))
    p1 = project_cam(a_cam)
    p2 = project_cam(b_cam)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    if label is not None:
        cv2.putText(img, label, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_base_marker(img, base_pos, color, label):
    """Big crosshair + filled disc at base_pos."""
    p = project_pts(base_pos.reshape(1, 3)).astype(int).reshape(2)
    px, py = int(p[0]), int(p[1])
    cv2.circle(img, (px, py), 10, color, -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 14, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (px - 25, py), (px + 25, py), color, 2, cv2.LINE_AA)
    cv2.line(img, (px, py - 25), (px, py + 25), color, 2, cv2.LINE_AA)
    cv2.putText(img, label, (px + 16, py - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# Infinite world axes (-100..+100m) through origin
draw_infinite_axis(comp, [1, 0, 0], (0, 0, 255), 1, label="+x")
draw_infinite_axis(comp, [0, 1, 0], (0, 255, 0), 1, label="+y")
draw_infinite_axis(comp, [0, 0, 1], (255, 0, 0), 1, label="+z")

# World origin axes (thick arrows) for emphasis
draw_axes(comp, np.zeros(3), np.eye(3), 0.20, 3)

# Base markers — magenta = arm1, cyan = arm2
draw_base_marker(comp, B1, (255, 0, 255), "B1")
draw_base_marker(comp, B2, (255, 255, 0), "B2")

# Target gripper frame (Vive) — what the visible gripper SHOULD look like
draw_axes(comp, p1_vive, R1_vive, 0.08, 3, labels=["g1x*", "g1y*", "g1z*"])
draw_axes(comp, p2_vive, R2_vive, 0.08, 3, labels=["g2x*", "g2y*", "g2z*"])
# Actual visible-gripper frame: link6 + GFIX_OFFSET (rotated by link6's R) for
# position, and link6 rot * Rx(180°) for visual rotation.
R_l1 = data.site_xmat[y1_ee].reshape(3, 3).copy()
R_l2 = data.site_xmat[y2_ee].reshape(3, 3).copy()
P_l1 = data.site_xpos[y1_ee].copy()
P_l2 = data.site_xpos[y2_ee].copy()
draw_axes(comp, P_l1 - R_l1 @ GFIX_OFFSET, R_l1 @ R_GFIX, 0.08, 1,
          labels=["g1x", "g1y", "g1z"])
draw_axes(comp, P_l2 - R_l2 @ GFIX_OFFSET, R_l2 @ R_GFIX, 0.08, 1,
          labels=["g2x", "g2y", "g2z"])

# Base axes (10cm arrows) — same colour scheme x=red y=green z=blue
R_base = np.column_stack([base_x, base_y, WORLD_UP])
draw_axes(comp, B1, R_base, 0.10, 2, labels=["b1x", "b1y", "b1z"])
draw_axes(comp, B2, R_base, 0.10, 2, labels=["b2x", "b2y", "b2z"])

cv2.putText(comp, f"frame {FRAME}  back={BACK}m down={DOWN}m",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
out_comp = ROOT / "data" / "inspect" / f"full_f{FRAME}_back{BACK:.2f}_down{DOWN:.2f}_comp.png"
cv2.imwrite(str(out_comp), comp)
print(f"\nwrote {out_raw}")
print(f"wrote {out_comp}")
