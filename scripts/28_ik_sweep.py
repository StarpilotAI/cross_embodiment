"""IK warm-start sweep video.

Starts both YAM arms in a neutral pose (whatever the seed `ready` joint config
produces), then over N frames linearly interpolates pos + slerps orientation
from that neutral pose to the target pose at FRAME. At each step we IK with
the previous solution as the warm start. If the IK can solve the final pose
gradually, the arm should arrive at the target without folding; if it gets
stuck, we'll see exactly where.

Usage:
  .venv/Scripts/python.exe scripts/28_ik_sweep.py 376 60
                                                 frame  n_steps
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 376
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 60
BASE_YAW_DEG  = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
HOLD          = int(sys.argv[4])   if len(sys.argv) > 4 else 80
BASE_TILT_DEG = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0   # forward tilt
BACK  = 0.50
DOWN  = 0.10
CAL_FRAME = 94

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


def quat_R(q):
    qx, qy, qz, qw = q
    n = (qx*qx+qy*qy+qz*qz+qw*qw) ** 0.5
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


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


def R_to_quat_wxyz(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        S = (tr+1.0)**0.5*2
        return np.array([0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = (1+R[0,0]-R[1,1]-R[2,2])**0.5*2
        return np.array([(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S])
    if R[1,1] > R[2,2]:
        S = (1+R[1,1]-R[0,0]-R[2,2])**0.5*2
        return np.array([(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S])
    S = (1+R[2,2]-R[0,0]-R[1,1])**0.5*2
    return np.array([(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S])


def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = (w*w+x*x+y*y+z*z) ** 0.5
    if n < 1e-9: return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def slerp(q0, q1, t):
    """Slerp two wxyz quaternions."""
    d = float(np.dot(q0, q1))
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        r = q0 + t*(q1-q0); return r / np.linalg.norm(r)
    th = np.arccos(d)
    s = np.sin(th)
    return (np.sin((1-t)*th)/s)*q0 + (np.sin(t*th)/s)*q1


# --- floor frame from CAL_FRAME ---
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
# Yaw around WORLD_UP
_th = np.radians(BASE_YAW_DEG)
base_x1 = np.cos(_th) * base_x0 + np.sin(_th) * base_y0
base_y1 = -np.sin(_th) * base_x0 + np.cos(_th) * base_y0
base_z1 = WORLD_UP
# Tilt around base_x1 (forward tilt: rotates base_z1 toward base_y1)
_ph = np.radians(BASE_TILT_DEG)
base_x = base_x1
base_y = np.cos(_ph) * base_y1 - np.sin(_ph) * base_z1
base_z = np.sin(_ph) * base_y1 + np.cos(_ph) * base_z1
base_xyaxes = " ".join(f"{v:.10f}" for v in list(base_x) + list(base_y))
print(f"BASE_YAW = {BASE_YAW_DEG}deg  BASE_TILT = {BASE_TILT_DEG}deg (forward)")
print(f"base_x = {base_x}")
print(f"base_y = {base_y}")
print(f"base_z = {base_z}")
B1 = p1_cal + BACK * BEHIND_DIR + DOWN * (-WORLD_UP)
B2 = p2_cal + BACK * BEHIND_DIR + DOWN * (-WORLD_UP)

# --- final target ---
# Canonical mounting: gripper body sits at link6 origin with quat=I.
# To replicate script 21 (mocap=Vive with internal gfix at -0.1092z + Rx(180°)):
#   P_link6 = P_vive + R_vive @ (0, 0, -0.1092)
#   R_link6 = R_vive @ Rx(180°)
R_GFIX = np.diag([1.0, -1.0, -1.0])
GFIX_OFFSET = np.array([0.0, 0.0, -0.1092])
s = META[FRAME]["state"]
P1_vive, R1_vive = gripper_in_cal(s, 1)
P2_vive, R2_vive = gripper_in_cal(s, 2)
P1_f = P1_vive + R1_vive @ GFIX_OFFSET; R1_f = R1_vive @ R_GFIX
P2_f = P2_vive + R2_vive @ GFIX_OFFSET; R2_f = R2_vive @ R_GFIX

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
                  <!-- Canonical i2rt linear_4310 mounting: gripper body has
                       pos=0,0,0 quat=identity inside link6. No fixup. -->
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
<mujoco model="yam_full_sweep">
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
xml_path = ROOT / "data" / "yam_sweep.xml"
xml_path.write_text(SCENE_XML)
model = MjModel.from_xml_path(str(xml_path))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)

y1_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1,7)]
y2_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y2_j{j}") for j in range(1,7)]
y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt]
y2_qadr = [model.jnt_qposadr[j] for j in y2_jnt]
y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt]
y2_dofadr = [model.jnt_dofadr[j] for j in y2_jnt]
y1_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
y2_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y2_ee")

# Seed both arms with the ready pose, forward-kinematic to get the neutral
# (P, R) at link6.  These are our IK starting targets.
ready = np.array([0.0, 1.2, 1.2, 0.0, -0.3, 0.0])
for i, a in enumerate(y1_qadr): data.qpos[a] = ready[i]
for i, a in enumerate(y2_qadr): data.qpos[a] = ready[i]
mujoco.mj_forward(model, data)
P1_0 = data.site_xpos[y1_ee].copy()
R1_0 = data.site_xmat[y1_ee].reshape(3,3).copy()
P2_0 = data.site_xpos[y2_ee].copy()
R2_0 = data.site_xmat[y2_ee].reshape(3,3).copy()
q1_0_wxyz = R_to_quat_wxyz(R1_0); q1_f_wxyz = R_to_quat_wxyz(R1_f)
q2_0_wxyz = R_to_quat_wxyz(R2_0); q2_f_wxyz = R_to_quat_wxyz(R2_f)
print(f"arm1 neutral: pos={P1_0} -> target pos={P1_f}")
print(f"arm2 neutral: pos={P2_0} -> target pos={P2_f}")


def ik_full(qadr, dofadr, jnt_ids, site, target_pos, target_R,
            n_iter_pos=80, n_iter_full=200, rot_weight=0.5):
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


def project_pts(pts_world):
    pts = np.asarray(pts_world, np.float64).reshape(-1, 3)
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
    return proj.reshape(-1, 2)

def draw_axes(img, origin, R_frame, length, thickness, labels=None):
    o = np.asarray(origin, np.float64).reshape(1, 3)
    pts = np.vstack([o,
                     o + length * R_frame[:, 0],
                     o + length * R_frame[:, 1],
                     o + length * R_frame[:, 2]])
    p = project_pts(pts).astype(int)
    o2 = tuple(p[0])
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    names = labels or ["x","y","z"]
    for i in range(3):
        cv2.arrowedLine(img, o2, tuple(p[1+i]), colors[i], thickness, tipLength=0.18)
        cv2.putText(img, names[i], tuple(p[1+i]+np.array([4,-4])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1, cv2.LINE_AA)

bg = cv2.imread(str(ROOT / "artifacts" / "pinhole_inpainted" / f"{FRAME:06d}.png"))

out_dir = ROOT / "data" / "inspect" / "ik_sweep"
out_dir.mkdir(parents=True, exist_ok=True)
# Clear out stale frames so a shorter run doesn't leave old PNGs in the dir
for p in out_dir.glob("*.png"):
    p.unlink()
total = N + 1 + HOLD
print(f"\nrendering {N+1} interp steps + {HOLD} hold frames -> {out_dir}/")
for step in range(total):
    t = min(step, N) / N  # 0..1 then stays at 1 for HOLD frames
    # Reset arms to ready, then drive IK to interpolated targets
    if step == 0:
        for i, a in enumerate(y1_qadr): data.qpos[a] = ready[i]
        for i, a in enumerate(y2_qadr): data.qpos[a] = ready[i]
    P1_t = (1-t) * P1_0 + t * P1_f
    P2_t = (1-t) * P2_0 + t * P2_f
    R1_t = quat_wxyz_to_R(slerp(q1_0_wxyz, q1_f_wxyz, t))
    R2_t = quat_wxyz_to_R(slerp(q2_0_wxyz, q2_f_wxyz, t))
    # Vive-space target (where the VISIBLE gripper should be) for overlay only
    P1_vive_t = P1_t - R1_t @ R_GFIX @ GFIX_OFFSET    # invert the gfix offset
    P2_vive_t = P2_t - R2_t @ R_GFIX @ GFIX_OFFSET
    R1_vive_t = R1_t @ R_GFIX                          # undo Rx(180°)
    R2_vive_t = R2_t @ R_GFIX
    ik_full(y1_qadr, y1_dofadr, y1_jnt, y1_ee, P1_t, R1_t)
    ik_full(y2_qadr, y2_dofadr, y2_jnt, y2_ee, P2_t, R2_t)
    mujoco.mj_forward(model, data)
    cp1 = data.site_xpos[y1_ee].copy(); cR1 = data.site_xmat[y1_ee].reshape(3,3).copy()
    cp2 = data.site_xpos[y2_ee].copy(); cR2 = data.site_xmat[y2_ee].reshape(3,3).copy()
    pe1 = np.linalg.norm(cp1 - P1_t) * 1000
    pe2 = np.linalg.norm(cp2 - P2_t) * 1000
    def rerr(Rc, Rt):
        tr = np.clip((np.trace(Rt@Rc.T)-1)/2, -1, 1)
        return float(np.degrees(np.arccos(tr)))
    re1 = rerr(cR1, R1_t); re2 = rerr(cR2, R2_t)
    renderer.update_scene(data, camera="scene")
    rgb = renderer.render()
    fg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    mask = (luma > 8).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(mask, (5,5), 0).astype(np.float32)/255.0
    a = np.stack([alpha]*3, axis=-1)
    comp = (a * fg + (1-a) * bg).astype(np.uint8)
    # Visible gripper target (Vive frame, thick) + actual visible gripper (thin)
    draw_axes(comp, P1_vive_t, R1_vive_t, 0.08, 3)
    draw_axes(comp, P2_vive_t, R2_vive_t, 0.08, 3)
    # Actual visible gripper = link6 + offset, link6_rot @ Rx(180°)
    cR1_v = cR1 @ R_GFIX
    cR2_v = cR2 @ R_GFIX
    cp1_v = cp1 - cR1 @ GFIX_OFFSET
    cp2_v = cp2 - cR2 @ GFIX_OFFSET
    draw_axes(comp, cp1_v, cR1_v, 0.08, 1)
    draw_axes(comp, cp2_v, cR2_v, 0.08, 1)
    label = f"step {step:3d}/{total-1}  t={t:.2f}" + ("  HOLD" if step > N else "")
    cv2.putText(comp, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(comp, f"arm1 pos {pe1:5.1f}mm rot {re1:5.1f}deg", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(comp, f"arm2 pos {pe2:5.1f}mm rot {re2:5.1f}deg", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{step:04d}.png"), comp)
    if step % 10 == 0 or step == total - 1:
        print(f"  step {step}/{total-1}  arm1 {pe1:.1f}mm {re1:.1f}deg  arm2 {pe2:.1f}mm {re2:.1f}deg")

print("done. writing mp4 ...")
import subprocess
mp4 = ROOT / "data" / "inspect" / f"ik_sweep_f{FRAME}.mp4"
subprocess.run(["ffmpeg", "-y", "-framerate", "20",
                "-i", str(out_dir / "%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(mp4)], check=True)
print(f"wrote {mp4}")
