"""Single-frame iteration script for base placement.

Usage:
  python scripts/25_render_one_frame.py            # uses defaults
  python scripts/25_render_one_frame.py 94 0.5     # frame=94, behind_dist=0.5m

Computes YAM base positions = gripper_pos + behind_dist * R_gripper @ [0,0,1]
at the chosen frame. Renders + composites just that one frame, with overlay
markers showing where the calibration says each gripper is. Run in <5s for
fast iteration on placement.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 94
BEHIND = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

CALIB = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
META  = json.loads((ROOT / "data" / "frames_meta.json").read_text())

K = np.array(CALIB["K"], dtype=np.float64)
rvec = np.array(CALIB["rvec"], dtype=np.float64)
tvec = np.array(CALIB["tvec"], dtype=np.float64)
W, H = CALIB["image_size"]
R_a1_to_a2 = np.array(CALIB.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(CALIB.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
CAL_ARM = CALIB.get("calibrated_arm", 2)

R_w2c, _ = cv2.Rodrigues(rvec)
R_c2w = R_w2c.T
cam_pos = -R_c2w @ tvec
fovy_deg = float(np.degrees(2 * np.arctan2(H/2, K[1,1])))
mj_x = R_c2w[:, 0]
mj_y = -R_c2w[:, 1]
WORLD_UP = mj_y
IMAGE_LEFT = -mj_x

xyaxes = " ".join(f"{v:.10f}" for v in list(mj_x) + list(mj_y))
pos_str = " ".join(f"{v:.10f}" for v in cam_pos)

# Base orientation: local +z = WORLD_UP, local +x = IMAGE_LEFT
base_x = IMAGE_LEFT
base_y = np.cross(WORLD_UP, IMAGE_LEFT)
base_y = base_y / np.linalg.norm(base_y)
base_xyaxes = " ".join(f"{v:.10f}" for v in list(base_x) + list(base_y))

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

# Compute base positions at the chosen frame
s = META[FRAME]["state"]
assert s is not None, f"frame {FRAME} has no state"
tgt1 = gripper_target_in_cal(s, 1)
tgt2 = gripper_target_in_cal(s, 2)
assert tgt1 is not None and tgt2 is not None, f"missing pose at frame {FRAME}"

# Base = gripper + BEHIND * R @ [0,0,1]  (behind in gripper's local +z)
B1 = tgt1[0] + BEHIND * (tgt1[1] @ np.array([0, 0, 1]))
B2 = tgt2[0] + BEHIND * (tgt2[1] @ np.array([0, 0, 1]))
print(f"frame {FRAME}, behind={BEHIND}m")
print(f"  gripper1 at {tgt1[0]}, base1 at {B1}")
print(f"  gripper2 at {tgt2[0]}, base2 at {B2}")
print(f"  base1 distance to gripper1: {np.linalg.norm(B1 - tgt1[0]):.3f}m")

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
                <body name="{prefix}_link6" pos="0 0 0" quat="1 0 0 0">
                  <joint name="{prefix}_j6" pos="0 0 0" axis="0 0 1" range="-2.0944 2.0944"/>
                  <site name="{prefix}_ee" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
                  <body name="{prefix}_gfix" pos="0 0 -0.1092" euler="3.14159265 0 0">
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
<mujoco model="oneframe">
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
    <camera name="scene" pos="{pos_str}" xyaxes="{xyaxes}" fovy="{fovy_deg:.6f}"/>
    {yam_chain("y1", B1, base_xyaxes)}
    {yam_chain("y2", B2, base_xyaxes)}
  </worldbody>
</mujoco>
"""
(ROOT / "data" / "yam_oneframe.xml").write_text(SCENE_XML)
model = MjModel.from_xml_path(str(ROOT / "data" / "yam_oneframe.xml"))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)

y1_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1,7)]
y2_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y2_j{j}") for j in range(1,7)]
y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt_ids]
y2_qadr = [model.jnt_qposadr[j] for j in y2_jnt_ids]
y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt_ids]
y2_dofadr = [model.jnt_dofadr[j] for j in y2_jnt_ids]
y1_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
y2_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y2_ee")

ready = np.array([0.0, 1.0, 1.2, 0.0, -0.3, 0.0])
def set_arm(arm, q):
    qadr = y1_qadr if arm == 1 else y2_qadr
    for i, a in enumerate(qadr):
        data.qpos[a] = q[i]

def ik_pos(arm, target, n_iter=200):
    dofadr = y1_dofadr if arm == 1 else y2_dofadr
    qadr = y1_qadr if arm == 1 else y2_qadr
    jnt_ids = y1_jnt_ids if arm == 1 else y2_jnt_ids
    site = y1_ee if arm == 1 else y2_ee
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    for _ in range(n_iter):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[site]
        if np.linalg.norm(err) < 1e-4:
            return
        mujoco.mj_jacSite(model, data, Jp, Jr, site)
        J = Jp[:, dofadr]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        step = dq
        max_step = np.max(np.abs(step))
        if max_step > 0.2: step *= (0.2 / max_step)
        for i, a in enumerate(qadr):
            data.qpos[a] += step[i]
        for i, jid in enumerate(jnt_ids):
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)

set_arm(1, ready); set_arm(2, ready)
ik_pos(1, tgt1[0])
ik_pos(2, tgt2[0])
mujoco.mj_forward(model, data)
print(f"  ee1 err: {np.linalg.norm(data.site_xpos[y1_ee] - tgt1[0])*1000:.2f}mm")
print(f"  ee2 err: {np.linalg.norm(data.site_xpos[y2_ee] - tgt2[0])*1000:.2f}mm")

renderer.update_scene(data, camera="scene")
rgb = renderer.render()
render_path = ROOT / "data" / "inspect" / f"oneframe_f{FRAME}_b{BEHIND:.2f}.png"
cv2.imwrite(str(render_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Composite onto pinhole inpaint
bg = cv2.imread(str(ROOT / "artifacts" / "pinhole_inpainted" / f"{FRAME:06d}.png"))
fg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
mask = (luma > 8).astype(np.uint8) * 255
alpha = cv2.GaussianBlur(mask, (5,5), 0).astype(np.float32) / 255.0
a = np.stack([alpha]*3, axis=-1)
comp = (a * fg + (1-a) * bg).astype(np.uint8)
comp_path = ROOT / "data" / "inspect" / f"oneframe_comp_f{FRAME}_b{BEHIND:.2f}.png"
cv2.imwrite(str(comp_path), comp)
print(f"\nwrote {render_path}")
print(f"wrote {comp_path}")
