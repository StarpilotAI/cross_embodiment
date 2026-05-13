"""Sweep BASE_YAW_DEG at frame FRAME and report IK residuals.
Also tries per-arm mirroring (arm1 yaw=+a, arm2 yaw=-a).
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData

ROOT = Path(__file__).resolve().parent.parent
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 376
CAL_FRAME = 94
BACK = 0.50
DOWN = 0.10

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
R_GFIX = np.diag([1.0, -1.0, -1.0])

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
    R = quat_R(q)
    if vive_arm != CAL_ARM:
        if CAL_ARM == 1: Rt, tt = R_a2_to_a1, t_a2_to_a1
        else:            Rt, tt = R_a1_to_a2, t_a1_to_a2
        pos = Rt @ pos + tt
        R = Rt @ R
    return pos, R

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
B1 = p1_cal + BACK * BEHIND_DIR + DOWN * (-WORLD_UP)
B2 = p2_cal + BACK * BEHIND_DIR + DOWN * (-WORLD_UP)

s = META[FRAME]["state"]
P1_v, R1_v = gripper_in_cal(s, 1); R1_t = R1_v @ R_GFIX
P2_v, R2_v = gripper_in_cal(s, 2); R2_t = R2_v @ R_GFIX

YAM_ASSETS = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam" / "assets"

def yam_chain(prefix, base_pos, xyaxes):
    bx, by, bz = base_pos
    return f"""
    <body name="{prefix}_base" pos="{bx} {by} {bz}" xyaxes="{xyaxes}">
      <geom class="yam_dark" pos="-0.0374966 -0.0464005 0.187501" quat="0.707105 0 0.707108 0" mesh="base"/>
      <body name="{prefix}_link1" pos="0 0 0.067" quat="0.707105 0 0 -0.707108">
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
        <joint name="{prefix}_j1" pos="0 0 0" axis="0 0 1" range="-2.61799 3.05433"/>
        <geom class="yam_link" pos="-0.0464 0.0374968 0.119501" quat="0.499998 0.5 0.5 -0.500002" mesh="link1"/>
        <body name="{prefix}_link2" pos="-0.0329 0.02 0.0455" quat="0.499998 0.5 -0.500002 -0.5">
          <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
          <joint name="{prefix}_j2" pos="0 0 0" axis="0 0 1" range="0 3.65"/>
          <geom class="yam_link" pos="-0.0174977 -0.0740001 -0.07925" quat="0.499998 0.5 0.500002 0.5" mesh="link2"/>
          <body name="{prefix}_link3" pos="0.264 4.08431e-07 -0.06375" quat="9.38184e-07 -0.707105 -0.707108 9.38187e-07">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <joint name="{prefix}_j3" pos="0 0 0" axis="0 0 1" range="0 3.66519"/>
            <geom class="yam_link" pos="0.0740003 -0.281499 -0.0813" quat="9.38184e-07 9.38187e-07 -0.707108 -0.707105" mesh="link3"/>
            <body name="{prefix}_link4" pos="0.0600003 -0.244999 -0.00205" quat="1.32679e-06 0 0 -1">
              <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
              <joint name="{prefix}_j4" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
              <geom class="yam_link" pos="-0.0138003 0.0364989 -0.0787882" quat="0.707105 0.707108 0 0" mesh="link4"/>
              <body name="{prefix}_link5" pos="-0.0403003 0.0703851 -0.0323887" quat="9.38184e-07 -0.707105 -9.38187e-07 -0.707108">
                <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
                <joint name="{prefix}_j5" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
                <geom class="yam_dark" pos="-0.0463995 0.0311519 0.0265" quat="0.499998 -0.5 -0.5 -0.500002" mesh="link5"/>
                <body name="{prefix}_link6" pos="0 0 0" quat="1 0 0 0">
                  <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
                  <joint name="{prefix}_j6" pos="0 0 0" axis="0 0 1" range="-2.0944 2.0944"/>
                  <site name="{prefix}_ee" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
"""

def make_xml(y1_xyaxes, y2_xyaxes):
    return f"""
<mujoco model="probe">
  <compiler angle="radian" meshdir="{YAM_ASSETS.as_posix()}"/>
  <visual><global offwidth="{W}" offheight="{H}" fovy="{fovy_deg:.6f}"/></visual>
  <default>
    <default class="yam_link"><geom type="mesh"/></default>
    <default class="yam_dark"><geom type="mesh"/></default>
  </default>
  <asset>
    <mesh name="base"  file="base.stl"/>
    <mesh name="link1" file="link1.stl"/>
    <mesh name="link2" file="link2.stl"/>
    <mesh name="link3" file="link3.stl"/>
    <mesh name="link4" file="link4.stl"/>
    <mesh name="link5" file="link5.stl"/>
  </asset>
  <worldbody>
    {yam_chain("y1", B1, y1_xyaxes)}
    {yam_chain("y2", B2, y2_xyaxes)}
  </worldbody>
</mujoco>
"""

def yawed(base_x0, base_y0, deg):
    th = np.radians(deg)
    bx = np.cos(th) * base_x0 + np.sin(th) * base_y0
    by = -np.sin(th) * base_x0 + np.cos(th) * base_y0
    return " ".join(f"{v:.10f}" for v in list(bx) + list(by))

def ik(data, model, qadr, dofadr, jnt_ids, site, target_pos, target_R,
       n_iter_pos=300, n_iter_full=500, rot_weight=0.5):
    Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
    for _ in range(n_iter_pos):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site]
        if np.linalg.norm(err) < 1e-5: break
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

def evaluate(y1_yaw, y2_yaw):
    y1_x = yawed(base_x0, base_y0, y1_yaw)
    y2_x = yawed(base_x0, base_y0, y2_yaw)
    xml = make_xml(y1_x, y2_x)
    (ROOT/"data"/"probe.xml").write_text(xml)
    model = MjModel.from_xml_path(str(ROOT/"data"/"probe.xml"))
    data = MjData(model)
    y1_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1,7)]
    y2_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y2_j{j}") for j in range(1,7)]
    y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt]
    y2_qadr = [model.jnt_qposadr[j] for j in y2_jnt]
    y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt]
    y2_dofadr = [model.jnt_dofadr[j] for j in y2_jnt]
    y1_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
    y2_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y2_ee")
    best = None
    for seed in [[0,1.2,1.2,0,-0.3,0],
                 [0,0.6,0.6,0,0.3,0],
                 [0.5,1.5,0.5,0.5,-0.5,0.5],
                 [-0.5,1.0,1.5,-0.5,0.5,-0.5],
                 [1.0,1.0,1.0,0,0,0]]:
        for i, a in enumerate(y1_qadr): data.qpos[a] = seed[i]
        for i, a in enumerate(y2_qadr): data.qpos[a] = seed[i]
        ik(data, model, y1_qadr, y1_dofadr, y1_jnt, y1_ee, P1_v, R1_t)
        ik(data, model, y2_qadr, y2_dofadr, y2_jnt, y2_ee, P2_v, R2_t)
        mujoco.mj_forward(model, data)
        e1 = np.linalg.norm(data.site_xpos[y1_ee] - P1_v) * 1000
        e2 = np.linalg.norm(data.site_xpos[y2_ee] - P2_v) * 1000
        def re(Rc, Rt):
            tr = np.clip((np.trace(Rt@Rc.T)-1)/2, -1, 1); return float(np.degrees(np.arccos(tr)))
        r1 = re(data.site_xmat[y1_ee].reshape(3,3), R1_t)
        r2 = re(data.site_xmat[y2_ee].reshape(3,3), R2_t)
        score = (e1 + e2)/1000 + (r1 + r2)/100   # rough cost
        if best is None or score < best[0]:
            best = (score, e1, r1, e2, r2)
    return best

print(f"frame {FRAME}, sweeping yaws (rough seed search)...")
print(f"{'y1':>6} {'y2':>6}   {'arm1 pos':>10} {'rot':>8}   {'arm2 pos':>10} {'rot':>8}   total")
for yaw in [0, 30, 45, 60, 70, 80, 85, 90, 95, 100, 110, 135, -30, -60, -90]:
    s, e1, r1, e2, r2 = evaluate(yaw, yaw)
    print(f"{yaw:>6} {yaw:>6}   {e1:>8.1f}mm {r1:>6.1f}deg   {e2:>8.1f}mm {r2:>6.1f}deg   {e1+e2:.1f}mm {r1+r2:.1f}deg")
print("\nmirrored (y1 +a, y2 -a):")
for yaw in [60, 70, 80, 90]:
    s, e1, r1, e2, r2 = evaluate(yaw, -yaw)
    print(f"{yaw:>6} {-yaw:>6}   {e1:>8.1f}mm {r1:>6.1f}deg   {e2:>8.1f}mm {r2:>6.1f}deg   {e1+e2:.1f}mm {r1+r2:.1f}deg")
