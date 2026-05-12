"""Render 3 candidate YAM base placements at frame 94.

User feedback: the two grippers are on the floor side-by-side at frame 94.
Take that line (g1 -> g2) as one floor axis. Make 3 guesses for "up" and
generate one image per option. Then user picks.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
FRAME = 94
BEHIND = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
DOWN   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

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
    n = (qx*qx+qy*qy+qz*qz+qw*qw) ** 0.5
    if n < 1e-9: return np.eye(3)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])

def gripper_in_cal(s, vive_arm):
    if vive_arm == 1:
        pos = np.array(s[7:10]); q = s[10:14]
    else:
        pos = np.array(s[23:26]); q = s[26:30]
    R = quat_R(q)
    if vive_arm != CAL_ARM:
        if CAL_ARM == 1:
            Rt, tt = R_a2_to_a1, t_a2_to_a1
        else:
            Rt, tt = R_a1_to_a2, t_a1_to_a2
        pos = Rt @ pos + tt
        R = Rt @ R
    return pos, R

s = META[FRAME]["state"]
p1, R1 = gripper_in_cal(s, 1)
p2, R2 = gripper_in_cal(s, 2)
# IK target = Vive pose directly. The fixup body inside link6 handles the
# 180-around-x convention, exactly mirroring script 21's gripper-only render.
R1_target = R1
R2_target = R2
print(f"frame {FRAME}:")
print(f"  gripper1 (cal): {p1}")
print(f"  gripper2 (cal): {p2}")

# Side-by-side axis (from g1 to g2)
D = p2 - p1
D_norm = np.linalg.norm(D)
if D_norm < 1e-3:
    # If the two grippers coincide at frame 94, fall back to a default
    print("  WARN: grippers coincide, using default x axis")
    D = np.array([0.3, 0, 0])
    D_norm = np.linalg.norm(D)
side_axis = D / D_norm
print(f"  side-by-side axis (cal): {side_axis}  (separation = {D_norm:.3f}m)")

# Gripper's local +z direction in world at frame 94 — this is presumably
# either the "back" or the "forward" direction depending on Vive convention.
# Average from arm1 and arm2 quats to be robust.
fwd_guess_a = R1[:, 2]  # Vive +z direction in world at frame 94
fwd_guess_b = -R1[:, 2]
print(f"  R1 @ +z (Vive forward candidate A): {fwd_guess_a}")

# 3 candidates for "up" direction
candidates = {
    "A_up=R@+z": fwd_guess_a,            # Vive's +z direction (cal frame)
    "B_up=cross_side_R+z": np.cross(side_axis, fwd_guess_a),  # perpendicular to both
    "C_up=cross_R+z_side": np.cross(fwd_guess_a, side_axis),  # opposite perp
}
# Normalize each
candidates = {k: v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else v
              for k, v in candidates.items()}

for k, v in candidates.items():
    print(f"  candidate {k}: UP = {v}")

# YAM XML chunks
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
                  <!-- Bright debug marker AT the BASE point so we can see it -->
                  <geom name="{prefix}_dbg_base" type="sphere" pos="0 0 -0.0546" size="0.04" rgba="0 1 0 1" contype="0" conaffinity="0"/>
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

def build_scene(label, world_up):
    """Generate scene XML with bases placed using `world_up`."""
    # Choose "forward" direction: the projection of fwd_guess_a onto the
    # floor plane (perpendicular to world_up)
    fwd = fwd_guess_a - np.dot(fwd_guess_a, world_up) * world_up
    fwd_n = np.linalg.norm(fwd)
    if fwd_n < 1e-3:
        # fallback: orthogonal to both side_axis and world_up
        fwd = np.cross(world_up, side_axis)
        fwd_n = np.linalg.norm(fwd)
    fwd = fwd / fwd_n
    behind = -fwd
    down = -world_up  # opposite of UP = downward
    # Base positions: each gripper offset behind + down
    B1 = p1 + BEHIND * behind + DOWN * down
    B2 = p2 + BEHIND * behind + DOWN * down
    print(f"  [{label}] world_up={world_up}, behind={behind}, B1={B1}, B2={B2}")

    # Base orientation: local +z = world_up, local +x = side_axis (consistent across both)
    # Make sure side_axis is perpendicular to world_up
    sx = side_axis - np.dot(side_axis, world_up) * world_up
    sx = sx / np.linalg.norm(sx)
    sy = np.cross(world_up, sx)
    sy = sy / np.linalg.norm(sy)
    # Validate: sx × sy ≈ world_up
    base_xyaxes = " ".join(f"{v:.10f}" for v in list(sx) + list(sy))

    SCENE_XML = f"""
<mujoco model="opt_{label}">
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
    return SCENE_XML

def render_option(label, world_up):
    xml = build_scene(label, world_up)
    xml_path = ROOT / "data" / f"opt_{label}.xml"
    xml_path.write_text(xml)
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

    ready = np.array([0.0, 1.0, 1.2, 0.0, -0.3, 0.0])
    for i, a in enumerate(y1_qadr): data.qpos[a] = ready[i]
    for i, a in enumerate(y2_qadr): data.qpos[a] = ready[i]

    def ik_full(qadr, dofadr, jnt_ids, site, target_pos, target_R, n_iter=200):
        Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
        # Stage A: position-only (converge to position first)
        for _ in range(n_iter):
            mujoco.mj_forward(model, data)
            err = target_pos - data.site_xpos[site]
            if np.linalg.norm(err) < 1e-4: break
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
        # Stage B: 6DoF — keep position AND match orientation
        for _ in range(n_iter):
            mujoco.mj_forward(model, data)
            cur_pos = data.site_xpos[site].copy()
            cur_R = data.site_xmat[site].reshape(3,3).copy()
            e_pos = target_pos - cur_pos
            R_err = target_R @ cur_R.T
            tr = np.trace(R_err)
            cos_a = np.clip((tr - 1) / 2, -1, 1)
            ang = np.arccos(cos_a)
            if ang > 1e-6:
                axis = np.array([R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
                axis = axis / (2 * np.sin(ang))
                e_rot = 0.1 * axis * ang   # low weight: position dominates
            else:
                e_rot = np.zeros(3)
            err = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(err) < 1e-4: break
            mujoco.mj_jacSite(model, data, Jp, Jr, site)
            J = np.vstack([Jp[:, dofadr], Jr[:, dofadr]])
            # Heavily damp orientation rows
            W = np.diag([1, 1, 1, 0.1, 0.1, 0.1])
            dq = J.T @ W @ np.linalg.solve(W @ J @ J.T @ W + 1e-2 * np.eye(6), W @ err)
            step = dq
            max_s = np.max(np.abs(step))
            if max_s > 0.1: step *= (0.1 / max_s)
            for i, a in enumerate(qadr): data.qpos[a] += step[i]
            for i, jid in enumerate(jnt_ids):
                lo, hi = model.jnt_range[jid]
                data.qpos[qadr[i]] = np.clip(data.qpos[qadr[i]], lo, hi)

    ik_full(y1_qadr, y1_dofadr, y1_jnt, y1_ee, p1, R1_target)
    ik_full(y2_qadr, y2_dofadr, y2_jnt, y2_ee, p2, R2_target)
    mujoco.mj_forward(model, data)
    err1 = np.linalg.norm(data.site_xpos[y1_ee] - p1) * 1000
    err2 = np.linalg.norm(data.site_xpos[y2_ee] - p2) * 1000
    print(f"    IK err: arm1={err1:.1f}mm arm2={err2:.1f}mm")

    renderer.update_scene(data, camera="scene")
    rgb = renderer.render()
    raw_path = ROOT / "data" / "inspect" / f"opt_{label}_raw.png"
    cv2.imwrite(str(raw_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Composite onto pinhole inpainted
    bg = cv2.imread(str(ROOT / "artifacts" / "pinhole_inpainted" / f"{FRAME:06d}.png"))
    fg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    luma = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    mask = (luma > 8).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(mask, (5,5), 0).astype(np.float32) / 255.0
    a = np.stack([alpha]*3, axis=-1)
    comp = (a * fg + (1-a) * bg).astype(np.uint8)
    # Label on the image
    cv2.putText(comp, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    comp_path = ROOT / "data" / "inspect" / f"opt_{label}_comp.png"
    cv2.imwrite(str(comp_path), comp)
    print(f"    wrote {comp_path}")
    return comp

for label, up in candidates.items():
    render_option(label, up)
print("done")
