"""Re-render YAM using the PnP-calibrated scene camera.

Reads:
  data/scene_cam_calibration.json   (PnP output: K, rvec, tvec)
  data/frames_meta.json
  data/yam_scene.xml                (Mujoco scene)

Writes:
  artifacts/render/*.png            (overwrites)

Mujoco's camera convention vs OpenCV:
  OpenCV cam: x→right, y→down, z→forward (looks in +z)
  Mujoco cam: x→right, y→up,   z→back    (looks in -z)
So Mujoco x = OpenCV x, Mujoco y = -OpenCV y, Mujoco z = -OpenCV z.

We DON'T need to keep the camera defined in the XML — we can override the
free camera per-frame from Python.
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer, MjvCamera

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "yam_scene.xml"
CALIB = json.loads((ROOT / "data" / "scene_cam_calibration.json").read_text())
META  = json.loads((ROOT / "data" / "frames_meta.json").read_text())
OUT_RGB = ROOT / "artifacts" / "render"
OUT_RGB.mkdir(parents=True, exist_ok=True)

K     = np.array(CALIB["K"], dtype=np.float64)
rvec  = np.array(CALIB["rvec"], dtype=np.float64)
tvec  = np.array(CALIB["tvec"], dtype=np.float64)
W, H  = CALIB["image_size"]
print(f"loaded calibration: K=\n{K}\nrvec={rvec}\ntvec={tvec}")

# OpenCV cam → world transform
R_w2c, _ = cv2.Rodrigues(rvec)
R_c2w = R_w2c.T
cam_pos_world = -R_c2w @ tvec

# Mujoco camera axes in world
mj_x = R_c2w[:, 0]            # right
mj_y = -R_c2w[:, 1]           # up  = -OpenCV y
print(f"cam pos (world): {cam_pos_world}")
print(f"mj cam x axis:   {mj_x}")
print(f"mj cam y axis:   {mj_y}")

# fovy from K (vertical full FOV in degrees)
fy = K[1, 1]
fovy_deg = float(np.degrees(2 * np.arctan2(H/2, fy)))
print(f"fovy = {fovy_deg:.2f} deg")

model = MjModel.from_xml_path(str(SCENE))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)

# Override the scene camera. We use a FREE camera and set its position +
# orientation explicitly each frame (more reliable than editing the XML
# camera mid-run).
cam = MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
# Set initial lookat / distance / azim / elev from cam_pos_world + forward dir.
# Simpler: directly drive the camera via mjvCamera helpers.

# We'll set the scene camera by recomputing on each frame; for now, the
# cam state is fixed (the dataset's scene cam is rigidly mounted).
# Mujoco's free-camera state is parameterised as (lookat, distance, azimuth, elevation).
# Convert from cam_pos + look direction to those.
look_dir = -R_c2w[:, 2]            # camera's -z axis in OpenCV = view direction
lookat = cam_pos_world + look_dir   # 1m in front
# distance & angles
delta = cam_pos_world - lookat
distance = float(np.linalg.norm(delta))
azimuth   = float(np.degrees(np.arctan2(delta[1], delta[0])))
elevation = float(np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]))))
cam.lookat[:] = lookat
cam.distance  = distance
cam.azimuth   = azimuth
cam.elevation = elevation
# Apply fovy via the model's free-cam fovy:
model.vis.global_.fovy = fovy_deg

def grip_width_to_slide(w):
    return float(np.clip(w * 0.5, 0.0, 0.0475))

ready_arm = np.array([0.0, 1.2, 0.8, 0.0, -0.5, 0.0])
print(f"rendering {len(META)} frames ... nq={model.nq}")
t0 = time.time()
for i, m in enumerate(META):
    joints = np.zeros(model.nq)
    joints[0:6] = ready_arm
    joints[8:14] = ready_arm
    if m["state"] is not None:
        s = m["state"]
        arm1_xyz = np.array(s[7:10])
        arm2_xyz = np.clip(np.array(s[7+16:10+16]), -0.5, 0.5)
        joints[0]  =  0.7 * arm1_xyz[0]
        joints[1]  = 1.0 + 0.6 * arm1_xyz[1]
        joints[2]  = 1.5 + 0.4 * arm1_xyz[2]
        joints[4]  = 0.6 - 0.6 * arm1_xyz[2]
        joints[6]  = grip_width_to_slide(s[15])
        joints[7]  = grip_width_to_slide(s[15])
        joints[8]  = -0.7 * arm2_xyz[0]
        joints[9]  = 1.0 + 0.6 * arm2_xyz[1]
        joints[10] = 1.5 + 0.4 * arm2_xyz[2]
        joints[12] = 0.6 - 0.6 * arm2_xyz[2]
        joints[14] = grip_width_to_slide(s[31])
        joints[15] = grip_width_to_slide(s[31])
    data.qpos[:] = joints
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    cv2.imwrite(str(OUT_RGB / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(META)}  {time.time()-t0:.1f}s", flush=True)
print(f"done in {time.time()-t0:.1f}s")
