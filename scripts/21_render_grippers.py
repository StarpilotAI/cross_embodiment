"""Render only the YAM grippers (linear_4310), each pinned to its Vive
tracker pose through the calibration. No arm, no IK — this isolates the
calibration question: do the YAM gripper meshes land where the operator's
PIKA grippers are?

If yes, we add the YAM arm next via IK to the wrist body.
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer, MjvCamera

ROOT = Path(__file__).resolve().parent.parent
CALIB = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
META  = json.loads((ROOT / "data" / "frames_meta.json").read_text())
OUT_RGB = ROOT / "artifacts" / "render"
OUT_RGB.mkdir(parents=True, exist_ok=True)

K     = np.array(CALIB["K"], dtype=np.float64)
rvec  = np.array(CALIB["rvec"], dtype=np.float64)
tvec  = np.array(CALIB["tvec"], dtype=np.float64)
W, H  = CALIB["image_size"]
R_a1_to_a2 = np.array(CALIB.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(CALIB.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a1_to_a2.T @ t_a1_to_a2
CAL_ARM = CALIB.get("calibrated_arm", 2)
print(f"calibrated_arm = {CAL_ARM}")
print(f"image_size = {W}x{H}")

# Cam -> world (OpenCV convention: +x right, +y down, +z forward)
R_w2c, _ = cv2.Rodrigues(rvec)
R_c2w = R_w2c.T
cam_pos = -R_c2w @ tvec
fy = K[1, 1]
fovy_deg = float(np.degrees(2 * np.arctan2(H/2, fy)))
# Mujoco camera convention: +x right, +y UP, +z BACK (away from scene).
# So world axes for the mujoco cam:
#   mj_x = opencv +x (cam right)        =  R_c2w[:,0]
#   mj_y = -opencv +y (cam up = -down)  = -R_c2w[:,1]
mj_x_world = R_c2w[:, 0]
mj_y_world = -R_c2w[:, 1]
xyaxes_str = " ".join(f"{v:.10f}" for v in
                     list(mj_x_world) + list(mj_y_world))
pos_str    = " ".join(f"{v:.10f}" for v in cam_pos)
print(f"mj camera pos  = {cam_pos}")
print(f"mj camera xyaxes = ({mj_x_world}, {mj_y_world})")
print(f"mj camera fovy = {fovy_deg:.3f} deg")

# --- Build a tiny scene: two mocap grippers, neutral aluminium ---
gripper_assets = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310" / "assets"
SCENE_XML = f"""
<mujoco model="grippers_only">
  <compiler angle="radian" meshdir="{gripper_assets.as_posix()}"/>
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="{fovy_deg:.6f}"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <mesh name="gripper_body" file="gripper.stl"/>
    <mesh name="tip_left"     file="tip_left.stl"/>
    <mesh name="tip_right"    file="tip_right.stl"/>
    <material name="alum" rgba="0.78 0.78 0.80 1.0" specular="0.4" shininess="0.6"/>
    <material name="dark" rgba="0.20 0.20 0.22 1.0" specular="0.1" shininess="0.2"/>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="32" height="32"/>
  </asset>
  <worldbody>
    <!-- Calibrated scene camera. pos + xyaxes derived directly from OpenCV
         rvec/tvec/K so this exactly matches cv2.projectPoints. -->
    <camera name="scene" pos="{pos_str}" xyaxes="{xyaxes_str}" fovy="{fovy_deg:.6f}"/>
    <body name="g1" pos="0 0 0" mocap="true">
      <!-- Yellow dot at calibrated BASE point — should sit on the user's click -->
      <geom name="g1_base" type="sphere" pos="0 0 -0.0546" size="0.008" rgba="1 1 0 1" contype="0" conaffinity="0"/>
      <!-- Fixup body: Vive's +z direction maps to linear_4310's -z direction.
           Apply Rx(180°) to flip y/z, and shift -0.109m in mocap-z so that
           after the flip, the gripper BASE lands at (0,0,-0.0546) in mocap
           frame — exactly where our calibration says BASE should be. -->
      <body name="g1_fix" pos="0 0 -0.1092" euler="3.14159265 0 0">
        <geom pos="-0.014 -0.0463995 0.0731" quat="1 0 0 0" type="mesh" mesh="gripper_body" material="dark" contype="0" conaffinity="0"/>
        <body name="g1_tl" pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002">
          <geom pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="tip_left" material="alum" contype="0" conaffinity="0"/>
        </body>
        <body name="g1_tr" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">
          <geom pos="-0.0379932 0.129783 0.00133753" quat="0.707105 -0.707108 0 0" type="mesh" mesh="tip_right" material="alum" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="g2" pos="0 0 0" mocap="true">
      <geom name="g2_base" type="sphere" pos="0 0 -0.0546" size="0.008" rgba="1 1 0 1" contype="0" conaffinity="0"/>
      <body name="g2_fix" pos="0 0 -0.1092" euler="3.14159265 0 0">
        <geom pos="-0.014 -0.0463995 0.0731" quat="1 0 0 0" type="mesh" mesh="gripper_body" material="dark" contype="0" conaffinity="0"/>
        <body name="g2_tl" pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002">
          <geom pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="tip_left" material="alum" contype="0" conaffinity="0"/>
        </body>
        <body name="g2_tr" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">
          <geom pos="-0.0379932 0.129783 0.00133753" quat="0.707105 -0.707108 0 0" type="mesh" mesh="tip_right" material="alum" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
xml_path = ROOT / "data" / "yam_grippers_only.xml"
xml_path.write_text(SCENE_XML)

model = MjModel.from_xml_path(str(xml_path))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)

g1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g1")
g2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "g2")
g1_mc = model.body_mocapid[g1_id]
g2_mc = model.body_mocapid[g2_id]

# The "scene" camera is embedded in the XML with pose+fovy from calibration —
# no need to set anything else here.

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

def R_to_wxyz(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        S = (tr+1.0)**0.5*2
        return (0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S)
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = (1+R[0,0]-R[1,1]-R[2,2])**0.5*2
        return ((R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S)
    if R[1,1] > R[2,2]:
        S = (1+R[1,1]-R[0,0]-R[2,2])**0.5*2
        return ((R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S)
    S = (1+R[2,2]-R[0,0]-R[1,1])**0.5*2
    return ((R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S)

OFFSCREEN = np.array([0, -100, 0])

def vive_to_calframe(pos, q, vive_arm):
    """Transform (pos, R) from Vive's `vive_arm` frame into the calibrated frame.
    The calibrated camera lives in the frame of Vive's `CAL_ARM`.
    """
    R = quat_to_R(q)
    if vive_arm == CAL_ARM:
        return pos, R
    if CAL_ARM == 1:  # incoming arm is 2 → arm2->arm1
        Rt, tt = R_a2_to_a1, t_a2_to_a1
    else:             # CAL_ARM == 2, incoming is 1
        Rt, tt = R_a1_to_a2, t_a1_to_a2
    return Rt @ pos + tt, Rt @ R

# Vive frame -> linear_4310 frame correction. Identity by default — this
# matches scripts/22_gripper_overlay_cv2.py exactly so the mujoco render
# and the cv2-projected silhouette overlap perfectly.
R_GRIPPER_FIX = np.eye(3, dtype=np.float64)

print(f"rendering {len(META)} frames ...")
t0 = time.time()
for i, m in enumerate(META):
    # Default off-screen
    data.mocap_pos[g1_mc] = OFFSCREEN
    data.mocap_pos[g2_mc] = OFFSCREEN
    if m["state"] is not None:
        s = m["state"]
        p1 = np.array(s[7:10]); q1 = (s[10], s[11], s[12], s[13])
        p2 = np.array(s[23:26]); q2 = (s[26], s[27], s[28], s[29])
        if np.linalg.norm(p1) > 1e-6 and np.max(np.abs(p1)) < 1:
            P, R = vive_to_calframe(p1, q1, vive_arm=1)
            data.mocap_pos[g1_mc] = P
            data.mocap_quat[g1_mc] = R_to_wxyz(R @ R_GRIPPER_FIX)
        if np.linalg.norm(p2) > 1e-6 and np.max(np.abs(p2)) < 1:
            P, R = vive_to_calframe(p2, q2, vive_arm=2)
            data.mocap_pos[g2_mc] = P
            data.mocap_quat[g2_mc] = R_to_wxyz(R @ R_GRIPPER_FIX)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="scene")
    rgb = renderer.render()
    cv2.imwrite(str(OUT_RGB / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(META)}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s")
