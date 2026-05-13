"""Full bimanual YAM render with IK.

Builds a Mujoco scene with two YAM arms (link1..link6 + linear_4310 gripper
attached via the fixup body so its frame matches Vive's). For each frame:

  1. Target pose for each arm = calibrated Vive gripper pose
  2. Damped-least-squares IK on joints 1..6 to drive the "ee" site
     (placed at the gripper-attach point on link6) to the target pose
  3. Render through the calibrated XML camera

Output: artifacts/render/*.png  (overwrites)
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
CALIB = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
META  = json.loads((ROOT / "data" / "frames_meta.json").read_text())
OUT_RGB = ROOT / "artifacts" / "render"
OUT_RGB.mkdir(parents=True, exist_ok=True)

K = np.array(CALIB["K"], dtype=np.float64)
rvec = np.array(CALIB["rvec"], dtype=np.float64)
tvec = np.array(CALIB["tvec"], dtype=np.float64)
W, H = CALIB["image_size"]
R_a1_to_a2 = np.array(CALIB.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(CALIB.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
CAL_ARM = CALIB.get("calibrated_arm", 2)

# Cam pose (OpenCV)
R_w2c, _ = cv2.Rodrigues(rvec)
R_c2w = R_w2c.T
cam_pos = -R_c2w @ tvec
fovy_deg = float(np.degrees(2 * np.arctan2(H/2, K[1,1])))
# Mujoco camera xyaxes in world
mj_x = R_c2w[:, 0]
mj_y = -R_c2w[:, 1]
xyaxes = " ".join(f"{v:.10f}" for v in list(mj_x) + list(mj_y))
pos_str = " ".join(f"{v:.10f}" for v in cam_pos)

# YAM base placement (in calibrated cam frame).
# We derive world directions from the calibrated camera:
#   world_up   = camera's image-up direction in world (= -R_c2w[:,1])
#   image_left = direction from workspace toward the operator
#   side       = perpendicular to both, used for side-by-side spacing
WORKSPACE_CENTER = np.array([-0.17, 0.03, -0.09])
WORLD_UP = mj_y         # (0.78, 0.61, 0.15)
IMAGE_LEFT = -mj_x      # operator side
SIDE = np.cross(WORLD_UP, IMAGE_LEFT)
SIDE = SIDE / np.linalg.norm(SIDE)

# Bases mounted on a horizontal floor below the workspace, on the operator's
# side. Side-by-side along the SIDE axis.
D_DOWN  = 0.30  # m below workspace
D_LEFT  = 0.10  # m toward operator
D_SIDE  = 0.15  # half-distance between the two bases
base_center = WORKSPACE_CENTER + D_DOWN * (-WORLD_UP) + D_LEFT * IMAGE_LEFT
YAM1_BASE = base_center + D_SIDE * SIDE
YAM2_BASE = base_center - D_SIDE * SIDE
print(f"YAM1_BASE = {YAM1_BASE}")
print(f"YAM2_BASE = {YAM2_BASE}")
print(f"WORLD_UP  = {WORLD_UP}")
# Base orientation: local +z = WORLD_UP, local +x = IMAGE_LEFT
# (so the arm extends 'up' and the natural forward direction points toward the operator)
base_x_world = IMAGE_LEFT
base_y_world = np.cross(WORLD_UP, IMAGE_LEFT)  # = -SIDE; cross product completes RHS
base_y_world = base_y_world / np.linalg.norm(base_y_world)
# Verify: base_x x base_y should equal WORLD_UP
check_z = np.cross(base_x_world, base_y_world)
print(f"base z-axis check (should be ~WORLD_UP): {check_z}")
base_xyaxes_str = " ".join(f"{v:.10f}" for v in list(base_x_world) + list(base_y_world))

# YAM mesh dir
YAM_ASSETS = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam" / "assets"
GRIP_ASSETS = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "assets"
# Copy gripper STLs into YAM asset folder (we did this earlier)

def yam_chain(prefix, base_pos, xyaxes):
    """Generate a full YAM kinematic chain ending in a fixup body
    so the gripper frame matches Vive. The 'ee' site is at link6 origin
    (which equals the gripper's mocap-pos point in script 21)."""
    bx, by, bz = base_pos
    return f"""
    <body name="{prefix}_base" pos="{bx} {by} {bz}" xyaxes="{xyaxes}">
      <geom class="yam_dark" pos="-0.0374966 -0.0464005 0.187501" quat="0.707105 0 0.707108 0" mesh="base"/>
      <body name="{prefix}_link1" pos="0 0 0.067" quat="0.707105 0 0 -0.707108">
        <inertial pos="-0.00261782 0.00744082 0.0237251" quat="0.602276 0.365837 0.642523 0.300984" mass="0.104154" diaginertia="0.000139023 0.000109628 4.8478e-05"/>
        <joint name="{prefix}_j1" pos="0 0 0" axis="0 0 1" range="-2.61799 3.05433"/>
        <geom class="yam_link" pos="-0.0464 0.0374968 0.119501" quat="0.499998 0.5 0.5 -0.500002" mesh="link1"/>
        <body name="{prefix}_link2" pos="-0.0329 0.02 0.0455" quat="0.499998 0.5 -0.500002 -0.5">
          <inertial pos="0.129165 0.000134495 -0.0321253" quat="0.499976 0.500008 0.500105 0.499911" mass="1.46908" diaginertia="0.0146686 0.014599 0.000800089"/>
          <joint name="{prefix}_j2" pos="0 0 0" axis="0 0 1" range="0 3.65"/>
          <geom class="yam_link" pos="-0.0174977 -0.0740001 -0.07925" quat="0.499998 0.5 0.500002 0.5" mesh="link2"/>
          <body name="{prefix}_link3" pos="0.264 4.08431e-07 -0.06375" quat="9.38184e-07 -0.707105 -0.707108 9.38187e-07">
            <inertial pos="0.0556042 -0.135083 -0.0340514" quat="0.704407 0.697077 0.121918 -0.0550519" mass="0.982553" diaginertia="0.00765198 0.0076296 0.000876442"/>
            <joint name="{prefix}_j3" pos="0 0 0" axis="0 0 1" range="0 3.66519"/>
            <geom class="yam_link" pos="0.0740003 -0.281499 -0.0813" quat="9.38184e-07 9.38187e-07 -0.707108 -0.707105" mesh="link3"/>
            <body name="{prefix}_link4" pos="0.0600003 -0.244999 -0.00205" quat="1.32679e-06 0 0 -1">
              <inertial pos="-0.0543529 0.057068 -0.0332231" quat="0.636075 0.627088 0.370444 0.254833" mass="0.46678" diaginertia="0.000817855 0.000791274 0.000295776"/>
              <joint name="{prefix}_j4" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
              <geom class="yam_link" pos="-0.0138003 0.0364989 -0.0787882" quat="0.707105 0.707108 0 0" mesh="link4"/>
              <body name="{prefix}_link5" pos="-0.0403003 0.0703851 -0.0323887" quat="9.38184e-07 -0.707105 -9.38187e-07 -0.707108">
                <inertial pos="-3.55003e-05 -0.00717397 0.0375847" quat="0.911516 0.411243 -0.00302177 -0.00303654" mass="0.403307" diaginertia="0.000219794 0.000195686 0.000169647"/>
                <joint name="{prefix}_j5" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
                <geom class="yam_dark" pos="-0.0463995 0.0311519 0.0265" quat="0.499998 -0.5 -0.5 -0.500002" mesh="link5"/>
                <body name="{prefix}_link6" pos="2.39858e-07 -0.0419481 0.0404996" quat="0.499998 -0.5 -0.5 -0.500002">
                  <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
                  <joint name="{prefix}_j6" pos="0 0 0" axis="0 0 -1" range="-2.0944 2.0944"/>
                  <site name="{prefix}_ee" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
                  <!-- Canonical i2rt linear_4310 mounting (pos=0,0,0 quat=I,
                       no fixup); link6 mount pos/quat/axis from config/linear_4310.yml -->
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
<mujoco model="yam_full_calibrated">
  <compiler angle="radian" meshdir="{YAM_ASSETS.as_posix()}"/>
  <option timestep="0.002"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="{fovy_deg:.6f}"/>
    <quality shadowsize="2048"/>
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
    <camera name="scene" pos="{pos_str}" xyaxes="{xyaxes}" fovy="{fovy_deg:.6f}"/>
    {yam_chain("y1", YAM1_BASE, base_xyaxes_str)}
    {yam_chain("y2", YAM2_BASE, base_xyaxes_str)}
  </worldbody>
</mujoco>
"""
xml_path = ROOT / "data" / "yam_full_calibrated.xml"
xml_path.write_text(SCENE_XML)
print(f"wrote {xml_path}")

model = MjModel.from_xml_path(str(xml_path))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)
print(f"model nq={model.nq} njnt={model.njnt}")

# IK setup
y1_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1,7)]
y2_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y2_j{j}") for j in range(1,7)]
y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt_ids]
y2_qadr = [model.jnt_qposadr[j] for j in y2_jnt_ids]
y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt_ids]
y2_dofadr = [model.jnt_dofadr[j] for j in y2_jnt_ids]
y1_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
y2_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y2_ee")
print(f"y1 joints qadr={y1_qadr} dofadr={y1_dofadr}, ee_site={y1_ee_site}")

# Initial seed pose
ready = np.array([0.0, 1.0, 1.2, 0.0, -0.3, 0.0])

def set_arm_qpos(arm, q):
    qadr = y1_qadr if arm == 1 else y2_qadr
    for i, a in enumerate(qadr):
        data.qpos[a] = q[i]

def get_arm_qpos(arm):
    qadr = y1_qadr if arm == 1 else y2_qadr
    return np.array([data.qpos[a] for a in qadr])

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

def gripper_target_in_cal(s, vive_arm):
    if vive_arm == 1:
        pos = np.array(s[7:10]); q = s[10:14]
    else:
        pos = np.array(s[23:26]); q = s[26:30]
    if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
       or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
        return None
    R = quat_to_R(q)
    if vive_arm != CAL_ARM:
        if CAL_ARM == 1:
            Rt, tt = R_a2_to_a1, t_a2_to_a1
        else:
            Rt, tt = R_a1_to_a2, t_a1_to_a2
        pos = Rt @ pos + tt
        R = Rt @ R
    return pos, R

def ik_solve(arm, target_pos, target_R, n_iter=80, alpha=1.0, damping=1e-3,
             pos_weight=1.0, rot_weight=0.0):
    """Damped least squares IK on the 6 joints of `arm`, driving the ee site
    to (target_pos, target_R). Two-stage: position-only first, then full 6DoF."""
    dofadr = y1_dofadr if arm == 1 else y2_dofadr
    qadr = y1_qadr if arm == 1 else y2_qadr
    jnt_ids = y1_jnt_ids if arm == 1 else y2_jnt_ids
    site = y1_ee_site if arm == 1 else y2_ee_site
    nv = len(dofadr)
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))

    # ---- Stage A: position-only IK (cheaper, ensures EE lands on target) ----
    for it in range(n_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site].copy()
        e_pos = pos_weight * (target_pos - cur_pos)
        if np.linalg.norm(e_pos) < 1e-4:
            break
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = Jp[:, dofadr]                         # 3 x nv_arm
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), e_pos)
        # Limit step size
        step = alpha * dq
        max_step = np.max(np.abs(step))
        if max_step > 0.2:
            step *= (0.2 / max_step)
        for i, a in enumerate(qadr):
            data.qpos[a] += step[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)

    # Skip stage B if rot_weight == 0
    if rot_weight <= 0:
        return
    # ---- Stage B: full 6DoF IK (refines orientation while preserving pos) ----
    for it in range(n_iter):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site].copy()
        cur_R = data.site_xmat[site].reshape(3,3).copy()
        e_pos = pos_weight * (target_pos - cur_pos)
        R_err = target_R @ cur_R.T
        tr = np.trace(R_err)
        cos_a = np.clip((tr - 1) / 2, -1, 1)
        ang = np.arccos(cos_a)
        if ang > 1e-6:
            axis = np.array([R_err[2,1]-R_err[1,2],
                             R_err[0,2]-R_err[2,0],
                             R_err[1,0]-R_err[0,1]])
            axis = axis / (2 * np.sin(ang))
            e_rot = rot_weight * axis * ang
        else:
            e_rot = np.zeros(3)
        err = np.concatenate([e_pos, e_rot])
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = np.vstack([Jp[:, dofadr], Jr[:, dofadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        step = alpha * dq
        max_step = np.max(np.abs(step))
        if max_step > 0.2:
            step *= (0.2 / max_step)
        for i, a in enumerate(qadr):
            data.qpos[a] += step[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)

# Seed both arms with ready pose
set_arm_qpos(1, ready)
set_arm_qpos(2, ready)

print(f"rendering {len(META)} frames ...")
t0 = time.time()
last_q1 = ready.copy()
last_q2 = ready.copy()
ik_residuals = []
for i, m in enumerate(META):
    if m["state"] is None:
        # Use last known pose
        set_arm_qpos(1, last_q1); set_arm_qpos(2, last_q2)
    else:
        s = m["state"]
        for arm_id, last_q, vive_arm in [(1, last_q1, 1), (2, last_q2, 2)]:
            tgt = gripper_target_in_cal(s, vive_arm)
            if tgt is None:
                set_arm_qpos(arm_id, last_q); continue
            tp, tR = tgt
            # Warm start from previous solution
            set_arm_qpos(arm_id, last_q)
            ik_solve(arm_id, tp, tR)
            new_q = get_arm_qpos(arm_id)
            if arm_id == 1: last_q1 = new_q
            else:           last_q2 = new_q
            # Measure residual error after IK
            mujoco.mj_forward(model, data)
            site = y1_ee_site if arm_id == 1 else y2_ee_site
            cur = data.site_xpos[site].copy()
            ik_residuals.append(np.linalg.norm(cur - tp))
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="scene")
    rgb = renderer.render()
    cv2.imwrite(str(OUT_RGB / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(META)}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s")
if ik_residuals:
    r = np.array(ik_residuals) * 1000  # mm
    print(f"IK residual (mm): mean={r.mean():.1f} med={np.median(r):.1f} p90={np.percentile(r,90):.1f} max={r.max():.1f}")
