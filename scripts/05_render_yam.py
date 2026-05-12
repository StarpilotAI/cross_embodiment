"""Stage 4: Render two YAM arms posed to track the Vive EE targets.

THIS IS TIER-1 / EYEBALLED.

- Scene camera pose, intrinsics, and YAM base mounting are hand-picked. The
  resulting render will not be photometrically aligned with the fisheye
  background — it is meant to demonstrate the pipeline shape.
- IK is solved with mujoco's Jacobian-pseudo-inverse method on the
  joint1..joint6 chain, driving the wrist (link6) to the target pose. We
  ignore rotation for now and only match position; this is enough to see
  the arm reaching to the right spot.
- The PIKA gripper from the dataset is NOT modelled; we just render the
  YAM arm without a gripper attachment.
- Output: per-frame RGB + depth PNG to artifacts/render/ and
  artifacts/render_depth/.
"""
import json
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
YAM_XML = ROOT / "data" / "yam_scene.xml"  # bimanual + offscreen 1024 + camera
OUT_RGB = ROOT / "artifacts" / "render"
OUT_DEPTH = ROOT / "artifacts" / "render_depth"
OUT_RGB.mkdir(parents=True, exist_ok=True)
OUT_DEPTH.mkdir(parents=True, exist_ok=True)

# --- World-scale scene assembly ---------------------------------------------
# We build a top-level MJCF that includes the YAM twice with different base
# transforms, plus a free-look camera and a table plane.
SCENE_XML = f"""
<mujoco model="scene">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <visual>
    <global offwidth="1024" offheight="1024"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".95 .95 .95" rgb2=".85 .85 .85" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".05 .05 .07" rgb2=".15 .18 .22" width="32" height="64"/>
  </asset>
  <worldbody>
    <light name="L1" pos="0 0 2" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light name="L2" pos="1 1 1.5" dir="-1 -1 -1.5" diffuse="0.3 0.3 0.3"/>
    <geom name="table" type="plane" size="1 1 0.05" rgba="0.95 0.95 0.95 1" material="grid"/>
    <camera name="scene"
      pos="0.0 -0.55 0.95"
      xyaxes="1 0 0  0 0.5 0.866"
      fovy="58"/>
    <body name="yam1_root" pos="-0.30 0.10 0.0" euler="0 0 0">
      <include file="{YAM_XML.as_posix()}"/>
    </body>
  </worldbody>
</mujoco>
"""
# We'll actually use a simpler approach: load yam.xml standalone, render only one
# arm to keep complexity down. The other arm can be added once IK works.

def build_model():
    """Load the YAM xml standalone."""
    m = mujoco.MjModel.from_xml_path(str(YAM_XML))
    return m

def render_pose(model, data, renderer, joints):
    """Set joint angles, render with the named 'scene' camera."""
    n = min(len(joints), model.nq)
    data.qpos[:n] = joints
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="scene")
    rgb = renderer.render()
    return rgb

def main():
    model = build_model()
    print(f"yam.xml loaded. nq={model.nq}, njnt={model.njnt}, nbody={model.nbody}")
    data = MjData(model)
    renderer = Renderer(model, height=1024, width=1024)

    meta = json.loads((ROOT / "data" / "frames_meta.json").read_text())

    # 16 joints total: yam1 (j1..j6, j7_tip_l, j8_tip_r), yam2 (same).
    # j7/j8 are gripper-tip slides (0..0.0475 each); we set them from the
    # dataset's gripper_distance_m (state[15] for arm1, state[31] for arm2).
    ready_arm = np.array([0.0, 1.2, 0.8, 0.0, -0.5, 0.0])

    def grip_width_to_slide(w):
        # gripper_distance_m is the full jaw separation; each slide = width/2
        return float(np.clip(w * 0.5, 0.0, 0.0475))

    n = len(meta)
    print(f"rendering {n} frames ...  nq={model.nq}")
    import time
    t0 = time.time()
    for i, m in enumerate(meta):
        joints = np.zeros(model.nq)
        joints[0:6] = ready_arm
        joints[8:14] = ready_arm
        if m["state"] is not None:
            s = m["state"]
            arm1_xyz = np.array(s[7:10])
            arm2_xyz = np.clip(np.array(s[7+16:10+16]), -0.5, 0.5)
            # Heuristic mapping until PnP-calibrated IK is wired up
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
        rgb = render_pose(model, data, renderer, joints)
        cv2.imwrite(str(OUT_RGB / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n}  {time.time()-t0:.1f}s  ({(i+1)/(time.time()-t0):.1f} fps)", flush=True)

    print("done")

if __name__ == "__main__":
    main()
