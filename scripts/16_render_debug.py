"""Calibration-debug render.

Build a minimal Mujoco scene containing only:
  • a world-frame triad at origin (red=+x, green=+y, blue=+z, each 30 cm long)
  • a 10 cm wireframe cube at origin
  • a small sphere at each arm's Vive gripper position (per-frame)
  • the *path* of each arm's gripper across the whole episode (faint dots)

Render this through the PnP-calibrated camera and composite onto the
pinhole-inpainted background. If the calibration is good, the per-frame
spheres should land exactly where the operator's grippers are visible in
the dataset image.
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
OUT_RGB = ROOT / "artifacts" / "debug_render"
OUT_RGB.mkdir(parents=True, exist_ok=True)

K     = np.array(CALIB["K"], dtype=np.float64)
rvec  = np.array(CALIB["rvec"], dtype=np.float64)
tvec  = np.array(CALIB["tvec"], dtype=np.float64)
W, H  = CALIB["image_size"]

# arm1->arm2 frame transform (recovered by 19_solve_pnp_auto.py)
R_a1_to_a2 = np.array(CALIB.get("R_arm1_to_arm2", np.eye(3).tolist()), dtype=np.float64)
t_a1_to_a2 = np.array(CALIB.get("t_arm1_to_arm2", np.zeros(3).tolist()), dtype=np.float64)
# Which arm the scene camera is calibrated to (1 or 2)
CAL_ARM = CALIB.get("calibrated_arm", 2)
print(f"calibrated_arm = {CAL_ARM}")
print(f"R_a1_to_a2 =\n{R_a1_to_a2}\nt_a1_to_a2 = {t_a1_to_a2}")

# Cam → world transform (OpenCV)
R_w2c, _ = cv2.Rodrigues(rvec)
R_c2w = R_w2c.T
cam_pos = -R_c2w @ tvec
fy = K[1, 1]
fovy_deg = float(np.degrees(2 * np.arctan2(H/2, fy)))
print(f"cam pos world: {cam_pos}")
print(f"fovy_deg = {fovy_deg:.2f}")

# Gather all Vive positions for the "trail" markers
arm1_path, arm2_path = [], []
for m in META:
    if m["state"] is None: continue
    s = m["state"]
    p1 = np.array(s[7:10]); p2 = np.array(s[23:26])
    if np.linalg.norm(p1) > 1e-6 and np.max(np.abs(p1)) < 1: arm1_path.append(p1)
    if np.linalg.norm(p2) > 1e-6 and np.max(np.abs(p2)) < 1: arm2_path.append(p2)
arm1_path = np.array(arm1_path); arm2_path = np.array(arm2_path)
print(f"arm1 path points: {len(arm1_path)}")
print(f"arm2 path points: {len(arm2_path)}")

# Subsample trail markers to keep XML small (~120 dots per arm)
def subsample(arr, n=120):
    if len(arr) <= n: return arr
    idx = np.linspace(0, len(arr)-1, n).astype(int)
    return arr[idx]
trail1 = subsample(arm1_path)
trail2 = subsample(arm2_path)

# --- Build debug scene XML at runtime ---
def axis_geom(name, axis_dir, color, length=0.30, radius=0.005):
    """Cylinder from origin extending `length` along axis_dir."""
    # Mujoco capsule between fromto endpoints
    fx, fy_, fz = 0, 0, 0
    tx, ty, tz = axis_dir[0]*length, axis_dir[1]*length, axis_dir[2]*length
    rgba = " ".join(map(str, color))
    return f'<geom name="{name}" type="capsule" fromto="{fx} {fy_} {fz} {tx} {ty} {tz}" size="{radius}" rgba="{rgba}" contype="0" conaffinity="0"/>'

def cube_at(pos, half=0.05, color=(0.9, 0.9, 0.0, 1.0)):
    rgba = " ".join(map(str, color))
    return f'<geom type="box" pos="{pos[0]} {pos[1]} {pos[2]}" size="{half} {half} {half}" rgba="{rgba}" contype="0" conaffinity="0"/>'

def sphere_at(pos, size=0.01, color=(1,0,0,1)):
    rgba = " ".join(map(str, color))
    return f'<geom type="sphere" pos="{pos[0]} {pos[1]} {pos[2]}" size="{size}" rgba="{rgba}" contype="0" conaffinity="0"/>'

trail_geoms = ""
for p in trail1:
    # arm1 lives in its own frame -> bring it into the (calibrated) arm2 frame
    p_a2 = R_a1_to_a2 @ p + t_a1_to_a2
    trail_geoms += sphere_at(p_a2, size=0.004, color=(0.2, 0.4, 1.0, 0.6)) + "\n"
for p in trail2:
    trail_geoms += sphere_at(p, size=0.004, color=(1.0, 0.5, 0.1, 0.6)) + "\n"

debug_xml = f"""
<mujoco model="debug">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <headlight diffuse="1 1 1" ambient="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="32" height="32"/>
  </asset>
  <worldbody>
    <!-- World axes at origin (30cm each) -->
    {axis_geom("axis_x", (1,0,0), (1.0, 0.1, 0.1, 1.0))}
    {axis_geom("axis_y", (0,1,0), (0.1, 1.0, 0.1, 1.0))}
    {axis_geom("axis_z", (0,0,1), (0.1, 0.3, 1.0, 1.0))}
    <!-- 10cm cube at origin -->
    {cube_at((0,0,0), half=0.05, color=(0.95, 0.85, 0.15, 0.6))}
    <!-- Per-arm marker bodies — driven per-frame via free joint -->
    <body name="arm1_marker" pos="0 0 0" mocap="true">
      <geom type="sphere" size="0.012" rgba="0.3 0.6 1 1" contype="0" conaffinity="0"/>
      {axis_geom("a1x", (1,0,0), (1.0, 0.1, 0.1, 1.0), length=0.10, radius=0.004)}
      {axis_geom("a1y", (0,1,0), (0.1, 1.0, 0.1, 1.0), length=0.10, radius=0.004)}
      {axis_geom("a1z", (0,0,1), (0.1, 0.3, 1.0, 1.0), length=0.10, radius=0.004)}
    </body>
    <body name="arm2_marker" pos="0 0 0" mocap="true">
      <geom type="sphere" size="0.012" rgba="1 0.6 0.3 1" contype="0" conaffinity="0"/>
      {axis_geom("a2x", (1,0,0), (1.0, 0.1, 0.1, 1.0), length=0.10, radius=0.004)}
      {axis_geom("a2y", (0,1,0), (0.1, 1.0, 0.1, 1.0), length=0.10, radius=0.004)}
      {axis_geom("a2z", (0,0,1), (0.1, 0.3, 1.0, 1.0), length=0.10, radius=0.004)}
    </body>
    <!-- Faint trail of gripper positions (static) -->
    {trail_geoms}
  </worldbody>
</mujoco>
"""
xml_path = ROOT / "data" / "yam_scene_debug.xml"
xml_path.write_text(debug_xml)
print(f"wrote {xml_path}")

model = MjModel.from_xml_path(str(xml_path))
data  = MjData(model)
renderer = Renderer(model, height=H, width=W)

# Find mocap body indices
a1_mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm1_marker")
a2_mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm2_marker")
a1_mc  = model.body_mocapid[a1_mid]
a2_mc  = model.body_mocapid[a2_mid]

# Calibrated free camera
cam = MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
look_dir = -R_c2w[:, 2]
lookat   = cam_pos + look_dir
delta    = cam_pos - lookat
cam.lookat[:] = lookat
cam.distance  = float(np.linalg.norm(delta))
cam.azimuth   = float(np.degrees(np.arctan2(delta[1], delta[0])))
cam.elevation = float(np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]))))
model.vis.global_.fovy = fovy_deg

print(f"rendering {len(META)} debug frames ...")
t0 = time.time()
def quat_xyzw_to_mj_wxyz(qx, qy, qz, qw):
    """Convert OpenCV-style (x,y,z,w) to Mujoco (w,x,y,z)."""
    n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
    if n < 1e-9: return (1.0, 0.0, 0.0, 0.0)
    return (qw/n, qx/n, qy/n, qz/n)

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

def R_to_quat_wxyz(R):
    # Mujoco mocap_quat is (w, x, y, z)
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = (1.0 + R[0,0] - R[1,1] - R[2,2]) ** 0.5 * 2
        w = (R[2,1] - R[1,2]) / S; x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S; z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = (1.0 + R[1,1] - R[0,0] - R[2,2]) ** 0.5 * 2
        w = (R[0,2] - R[2,0]) / S; x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S; z = (R[1,2] + R[2,1]) / S
    else:
        S = (1.0 + R[2,2] - R[0,0] - R[1,1]) ** 0.5 * 2
        w = (R[1,0] - R[0,1]) / S; x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S; z = 0.25 * S
    return (w, x, y, z)

for i, m in enumerate(META):
    # Update markers (position + orientation from Vive).
    # arm1 lives in its own frame -> transform into arm2 frame before
    # passing to the (arm2-calibrated) scene camera.
    if m["state"] is not None:
        s = m["state"]
        p1 = np.array(s[7:10]); q1 = (s[10], s[11], s[12], s[13])
        p2 = np.array(s[23:26]); q2 = (s[26], s[27], s[28], s[29])

        if np.linalg.norm(p1) > 1e-6 and np.max(np.abs(p1)) < 1:
            # Bring arm1 into the calibrated (arm2) world frame
            p1_in_a2 = R_a1_to_a2 @ p1 + t_a1_to_a2
            R1_in_a2 = R_a1_to_a2 @ quat_to_R(q1)
            data.mocap_pos[a1_mc] = p1_in_a2
            data.mocap_quat[a1_mc] = R_to_quat_wxyz(R1_in_a2)
        else:
            data.mocap_pos[a1_mc] = (0, -100, 0)

        if np.linalg.norm(p2) > 1e-6 and np.max(np.abs(p2)) < 1:
            data.mocap_pos[a2_mc] = p2
            data.mocap_quat[a2_mc] = quat_xyzw_to_mj_wxyz(*q2)
        else:
            data.mocap_pos[a2_mc] = (0, -100, 0)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    cv2.imwrite(str(OUT_RGB / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(META)}  {time.time()-t0:.1f}s")
print(f"done in {time.time()-t0:.1f}s")
