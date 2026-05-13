"""Full-episode bimanual YAM render with corrected link6 mount + IK.

Combines the proven pieces from this debug session:
  - Base placement from CAL_FRAME=94 (option B: world_up from floor-resting grippers)
  - +90deg base yaw + 0.5m back, 0.1m down
  - link6 mount overrides from data/i2rt/i2rt/robots/config/linear_4310.yml
  - Canonical i2rt gripper mounting (gripper body at link6 origin, no fixup)
  - IK target = (P_vive + R_vive @ (0,0,-0.1092), R_vive @ Rx(180)) so that the
    visible gripper geometry matches script 21's mocap render exactly

Outputs:
  artifacts/render/*.png            (full bimanual YAM, calibrated scene cam)
  artifacts/composite_pinhole/*.png (rendered YAM composited onto inpainted bg)
  artifacts/composite_pinhole.mp4   (final mp4)
"""
import json, sys, time, subprocess
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
CAL_FRAME = 94
BACK = 0.50
DOWN = 0.10
BASE_YAW_DEG = 90.0
ROLL_OFFSET_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0

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
mj_x = R_c2w[:, 0]; mj_y = -R_c2w[:, 1]
xyaxes_cam = " ".join(f"{v:.10f}" for v in list(mj_x) + list(mj_y))
pos_cam_str = " ".join(f"{v:.10f}" for v in cam_pos)

R_GFIX_BASE = np.diag([1.0, -1.0, -1.0])    # Rx(180°) — matches script 21 gfix
# Optional roll around the gripper's forward axis (link6 +z in local frame,
# = -Vive z in world). Applied AFTER Rx(180°), i.e. in link6's local frame.
_rad = np.radians(ROLL_OFFSET_DEG)
R_ROLL = np.array([[ np.cos(_rad), -np.sin(_rad), 0.0],
                   [ np.sin(_rad),  np.cos(_rad), 0.0],
                   [ 0.0,           0.0,          1.0]])
R_GFIX = R_GFIX_BASE @ R_ROLL
GFIX_OFFSET = np.array([0.0, 0.0, -0.1092])
print(f"ROLL_OFFSET_DEG = {ROLL_OFFSET_DEG}")


def quat_R(q):
    qx, qy, qz, qw = q
    n = (qx*qx+qy*qy+qz*qz+qw*qw) ** 0.5
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                     [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                     [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])


def gripper_in_cal(s, vive_arm):
    if vive_arm == 1:
        pos = np.array(s[7:10]); q = s[10:14]
    else:
        pos = np.array(s[23:26]); q = s[26:30]
    if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
       or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
        return None
    R = quat_R(q)
    if vive_arm != CAL_ARM:
        if CAL_ARM == 1: Rt, tt = R_a2_to_a1, t_a2_to_a1
        else:            Rt, tt = R_a1_to_a2, t_a1_to_a2
        pos = Rt @ pos + tt
        R = Rt @ R
    return pos, R


# Floor frame from CAL_FRAME
s_cal = META[CAL_FRAME]["state"]
p1_cal, R1_cal = gripper_in_cal(s_cal, 1)
p2_cal, R2_cal = gripper_in_cal(s_cal, 2)
side_axis = (p2_cal - p1_cal); side_axis /= np.linalg.norm(side_axis)
fwd_guess = R1_cal[:, 2]
WORLD_UP = np.cross(side_axis, fwd_guess); WORLD_UP /= np.linalg.norm(WORLD_UP)
fwd = fwd_guess - np.dot(fwd_guess, WORLD_UP) * WORLD_UP; fwd /= np.linalg.norm(fwd)
BEHIND_DIR = -fwd
base_x0 = side_axis - np.dot(side_axis, WORLD_UP) * WORLD_UP; base_x0 /= np.linalg.norm(base_x0)
base_y0 = np.cross(WORLD_UP, base_x0); base_y0 /= np.linalg.norm(base_y0)
_th = np.radians(BASE_YAW_DEG)
base_x = np.cos(_th) * base_x0 + np.sin(_th) * base_y0
base_y = -np.sin(_th) * base_x0 + np.cos(_th) * base_y0
base_xyaxes = " ".join(f"{v:.10f}" for v in list(base_x) + list(base_y))
B1 = p1_cal + BACK * BEHIND_DIR + DOWN * (-WORLD_UP)
B2 = p2_cal + BACK * BEHIND_DIR + DOWN * (-WORLD_UP)
print(f"B1={B1}\nB2={B2}\nWORLD_UP={WORLD_UP}\nBASE_YAW={BASE_YAW_DEG}deg")

YAM_ASSETS = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam" / "assets"


def yam_chain(prefix, base_pos, xyaxes):
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
                  <body name="{prefix}_gripper" pos="0 0 0" quat="1 0 0 0">
                    <inertial pos="0 0 0" mass="0.5" diaginertia="1e-4 1e-4 1e-4"/>
                    <geom pos="-0.014 -0.0463995 0.0731" quat="1 0 0 0" type="mesh" mesh="gripper_body" material="dark" contype="0" conaffinity="0"/>
                    <body name="{prefix}_tl" pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002">
                      <inertial pos="0 0 0" mass="0.07" diaginertia="6e-5 6e-5 3e-5"/>
                      <joint name="{prefix}_j7" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.0475"/>
                      <geom pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="tip_left" material="alum" contype="0" conaffinity="0"/>
                    </body>
                    <body name="{prefix}_tr" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">
                      <inertial pos="0 0 0" mass="0.07" diaginertia="6e-5 6e-5 3e-5"/>
                      <joint name="{prefix}_j8" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.0475"/>
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
<mujoco model="yam_full_episode">
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
xml_path = ROOT / "data" / "yam_full_episode.xml"
xml_path.write_text(SCENE_XML)
model = MjModel.from_xml_path(str(xml_path))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)
print(f"model nq={model.nq}")

y1_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1,7)]
y2_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y2_j{j}") for j in range(1,7)]
y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt]
y2_qadr = [model.jnt_qposadr[j] for j in y2_jnt]
y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt]
y2_dofadr = [model.jnt_dofadr[j] for j in y2_jnt]
y1_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
y2_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y2_ee")
# Gripper finger slide joints (j7 = tip_left, j8 = tip_right, both driven by width)
y1_j7 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "y1_j7")]
y1_j8 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "y1_j8")]
y2_j7 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "y2_j7")]
y2_j8 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "y2_j8")]
GRIPPER_STROKE = 0.0475   # per linear_4310.xml: tip slide range [0, 0.0475]

READY = np.array([0.0, 1.2, 1.2, 0.0, -0.3, 0.0])


def set_arm(arm, q):
    qadr = y1_qadr if arm == 1 else y2_qadr
    for i, a in enumerate(qadr): data.qpos[a] = q[i]


def get_arm(arm):
    qadr = y1_qadr if arm == 1 else y2_qadr
    return np.array([data.qpos[a] for a in qadr])


def ik_full(qadr, dofadr, jnt_ids, site, target_pos, target_R,
            n_iter_pos=200, n_iter_full=400, rot_weight=0.5):
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    for _ in range(n_iter_pos):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site]
        if np.linalg.norm(err) < 1e-4: break
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = Jp[:, dofadr]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        s2 = dq; m = np.max(np.abs(s2))
        if m > 0.2: s2 *= (0.2/m)
        for i, a in enumerate(qadr): data.qpos[a] += s2[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)
    for _ in range(n_iter_full):
        mujoco.mj_forward(model, data)
        cp = data.site_xpos[site].copy()
        cR = data.site_xmat[site].reshape(3,3).copy()
        e_pos = target_pos - cp
        R_err = target_R @ cR.T
        tr = np.clip((np.trace(R_err)-1)/2, -1, 1)
        ang = np.arccos(tr)
        if ang > 1e-6:
            sa = np.sin(ang)
            axis = np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]]) / (2*sa)
            e_rot = rot_weight * axis * ang
        else:
            e_rot = np.zeros(3)
        err = np.concatenate([e_pos, e_rot])
        if np.linalg.norm(err) < 1e-5: break
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = np.vstack([Jp[:, dofadr], rot_weight * Jr[:, dofadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        s2 = dq; m = np.max(np.abs(s2))
        if m > 0.1: s2 *= (0.1/m)
        for i, a in enumerate(qadr): data.qpos[a] += s2[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)


OUT_RENDER = ROOT / "artifacts" / "render"
OUT_COMP   = ROOT / "artifacts" / "composite_pinhole"
OUT_RENDER.mkdir(parents=True, exist_ok=True)
OUT_COMP.mkdir(parents=True, exist_ok=True)
PINHOLE_INPAINT = ROOT / "artifacts" / "pinhole_inpainted"

# Escape seeds tried when the warm-start IK leaves a large residual.
ESCAPE_SEEDS = [
    np.array([ 0.0, 1.2, 1.2,  0.0, -0.3,  0.0]),
    np.array([ 0.0, 0.6, 0.6,  0.0,  0.3,  0.0]),
    np.array([ 0.5, 1.5, 0.5,  0.5, -0.5,  0.5]),
    np.array([-0.5, 1.0, 1.5, -0.5,  0.5, -0.5]),
    np.array([ 1.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
    np.array([-1.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
    np.array([ 2.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
    np.array([-2.5, 1.0, 1.0,  0.0,  0.0,  0.0]),
]

ESCAPE_POS_MM   = 30.0   # if pos residual exceeds this after warm-start, retry
ESCAPE_ROT_DEG  = 15.0   # ... or if rot residual exceeds this


def project_pts(pts_world):
    pts = np.asarray(pts_world, np.float64).reshape(-1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
    return proj.reshape(-1, 2).astype(int)


def draw_target_axes(img, origin, R_frame, length=0.03):
    """Draw 3 small axis arrows + a center dot at the IK target on the image."""
    pts = np.vstack([origin,
                     origin + length * R_frame[:, 0],
                     origin + length * R_frame[:, 1],
                     origin + length * R_frame[:, 2]])
    p = project_pts(pts)
    o2 = tuple(p[0])
    for i, color in enumerate([(0,0,255), (0,255,0), (255,0,0)]):
        cv2.line(img, o2, tuple(p[1+i]), color, 1, cv2.LINE_AA)
    cv2.circle(img, o2, 3, (0, 255, 255), -1, cv2.LINE_AA)


set_arm(1, READY); set_arm(2, READY)
last_q1 = READY.copy()
last_q2 = READY.copy()
pos_residuals = []
rot_residuals = []
j6_values = []      # track j6 (gripper roll) across frames
print(f"\nrendering {len(META)} frames -> {OUT_RENDER}/ and {OUT_COMP}/")
t0 = time.time()
for i, m in enumerate(META):
    targets = []   # list of (P_vive, R_vive) to overlay on composite
    if m["state"] is None:
        set_arm(1, last_q1); set_arm(2, last_q2)
    else:
        s = m["state"]
        w1 = max(0.0, min(1.0, float(s[14]))) * GRIPPER_STROKE
        w2 = max(0.0, min(1.0, float(s[30]))) * GRIPPER_STROKE
        data.qpos[y1_j7] = w1; data.qpos[y1_j8] = w1
        data.qpos[y2_j7] = w2; data.qpos[y2_j8] = w2
        for arm_id, last_q in [(1, last_q1), (2, last_q2)]:
            tgt = gripper_in_cal(s, arm_id)
            if tgt is None:
                set_arm(arm_id, last_q); continue
            P_vive, R_vive = tgt
            P_target = P_vive + R_vive @ GFIX_OFFSET
            R_target = R_vive @ R_GFIX
            qadr   = y1_qadr   if arm_id == 1 else y2_qadr
            dofadr = y1_dofadr if arm_id == 1 else y2_dofadr
            jntids = y1_jnt    if arm_id == 1 else y2_jnt
            ee     = y1_ee     if arm_id == 1 else y2_ee

            # Pass 1: warm-start from previous frame's solution
            set_arm(arm_id, last_q)
            ik_full(qadr, dofadr, jntids, ee, P_target, R_target)
            mujoco.mj_forward(model, data)
            pe = np.linalg.norm(data.site_xpos[ee] - P_target) * 1000
            tr = np.clip((np.trace(R_target @ data.site_xmat[ee].reshape(3,3).T)-1)/2, -1, 1)
            re = float(np.degrees(np.arccos(tr)))
            best_q = get_arm(arm_id)
            best_score = pe + 5.0 * re   # 1 deg ~ 5 mm

            # Pass 2: if warm-start left a big residual, try escape seeds
            if pe > ESCAPE_POS_MM or re > ESCAPE_ROT_DEG:
                for seed in ESCAPE_SEEDS:
                    set_arm(arm_id, seed)
                    ik_full(qadr, dofadr, jntids, ee, P_target, R_target)
                    mujoco.mj_forward(model, data)
                    sp = np.linalg.norm(data.site_xpos[ee] - P_target) * 1000
                    str_ = np.clip((np.trace(R_target @ data.site_xmat[ee].reshape(3,3).T)-1)/2, -1, 1)
                    sr = float(np.degrees(np.arccos(str_)))
                    score = sp + 5.0 * sr
                    if score < best_score:
                        best_score = score; best_q = get_arm(arm_id)
                        pe, re = sp, sr
                # Restore the best joint config we found
                for k, a in enumerate(qadr): data.qpos[a] = best_q[k]
                mujoco.mj_forward(model, data)

            new_q = best_q
            if arm_id == 1: last_q1 = new_q
            else:           last_q2 = new_q
            pos_residuals.append(pe); rot_residuals.append(re)
            if arm_id == 1:
                j6_values.append(new_q[5])
            targets.append((P_vive, R_vive))
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="scene")
    rgb = renderer.render()
    fg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT_RENDER / f"{i:06d}.png"), fg)
    bg = cv2.imread(str(PINHOLE_INPAINT / f"{i:06d}.png"))
    if bg is not None:
        luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        mask = (luma > 8).astype(np.uint8) * 255
        alpha = cv2.GaussianBlur(mask, (5,5), 0).astype(np.float32)/255.0
        a = np.stack([alpha]*3, axis=-1)
        comp = (a * fg + (1-a) * bg).astype(np.uint8)
        for P_v, R_v in targets:
            draw_target_axes(comp, P_v, R_v)
        cv2.imwrite(str(OUT_COMP / f"{i:06d}.png"), comp)
    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed * (len(META)-i-1) / (i+1)
        print(f"  {i+1}/{len(META)}  {elapsed:.1f}s  eta {eta:.0f}s")

if pos_residuals:
    pr = np.array(pos_residuals); rr = np.array(rot_residuals)
    print(f"\nIK residual: pos mean={pr.mean():.2f}mm median={np.median(pr):.2f}mm p90={np.percentile(pr,90):.2f}mm max={pr.max():.1f}mm")
    print(f"             rot mean={rr.mean():.2f}deg median={np.median(rr):.2f}deg p90={np.percentile(rr,90):.2f}deg max={rr.max():.1f}deg")
if j6_values:
    j6 = np.array(j6_values)
    print(f"arm1 j6 (gripper roll): min={np.degrees(j6.min()):.1f}deg max={np.degrees(j6.max()):.1f}deg range={np.degrees(j6.max()-j6.min()):.1f}deg")

print("\nwriting mp4 ...")
mp4 = ROOT / "artifacts" / "composite_pinhole.mp4"
subprocess.run(["ffmpeg", "-y", "-framerate", "30",
                "-i", str(OUT_COMP / "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(mp4)], check=True)
print(f"wrote {mp4}")
