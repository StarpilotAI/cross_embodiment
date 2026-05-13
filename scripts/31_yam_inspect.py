"""Render the canonical i2rt YAM **with the linear_4310 gripper attached** at a
neutral pose from multiple angles. The canonical yam.xml itself does not include
a gripper (it expects combine_arm_and_gripper_xml() to splice one in at
runtime), so this script does that splice manually.
"""
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
W = H = 640
YAM_DIR = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam"
GRIP_DIR = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "gripper" / "linear_4310"

assets_dir = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam" / "assets"
abs_assets = assets_dir.as_posix()

GRIPPER_FRAGMENT = """
                <!-- linear_4310 gripper, mounted canonically (pos=0 0 0 quat=I) -->
                <body name="gripper" pos="0 0 0" quat="1 0 0 0">
                  <inertial pos="-3.57066e-05 0.000249371 -0.0293133" quat="0.667981 0.744113 -0.00433092 0.00882801" mass="0.553219" diaginertia="0.000409686 0.000369758 0.000324415"/>
                  <geom pos="-0.014 -0.0463995 0.0731" quat="1 0 0 0" type="mesh" rgba="0.11 0.29 0.012 1" mesh="gripper_mesh" contype="0" conaffinity="0"/>
                  <site name="tcp_site" pos="0 0 0" quat="0 1 0 0" size="0.005" rgba="1 0 0 1"/>
                  <site name="grasp_site" pos="0 0 -0.1347" quat="0 1 0 0" size="0.008" rgba="0 1 0 1"/>
                  <body name="tip_left" pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002">
                    <inertial pos="-0.0224716 0.0143408 -0.0426253" quat="0.498899 0.455348 -0.0576406 0.735143" mass="0.0710042" diaginertia="6.24193e-05 6.01079e-05 2.83591e-05"/>
                    <joint name="joint7" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.0475"/>
                    <geom pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5" type="mesh" rgba="0.6 0.79 0.92 1" mesh="tip_left" contype="0" conaffinity="0"/>
                  </body>
                  <body name="tip_right" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">
                    <inertial pos="-0.0143408 -0.0224716 -0.0426253" quat="0.281222 0.8726 0.16705 -0.362738" mass="0.0710042" diaginertia="6.24193e-05 6.01079e-05 2.83591e-05"/>
                    <joint name="joint8" pos="0 0 0" type="slide" axis="0 0 -1" range="0 0.0475"/>
                    <geom pos="-0.0379932 0.129783 0.00133753" quat="0.707105 -0.707108 0 0" type="mesh" rgba="0.6 0.79 0.92 1" mesh="tip_right" contype="0" conaffinity="0"/>
                  </body>
                </body>
"""

# Build the canonical yam.xml + camera + gripper splice
src = (YAM_DIR / "yam.xml").read_text()
src = src.replace(
    '<compiler angle="radian" meshdir="assets"/>',
    f'<compiler angle="radian" meshdir="{abs_assets}"/>'
)
src = src.replace(
    "<asset>",
    f"""<asset>
    <mesh name="gripper_mesh" file="{(GRIP_DIR/'assets'/'gripper.stl').as_posix()}"/>
    <mesh name="tip_left" file="{(GRIP_DIR/'assets'/'tip_left.stl').as_posix()}"/>
    <mesh name="tip_right" file="{(GRIP_DIR/'assets'/'tip_right.stl').as_posix()}"/>"""
)
src = src.replace(
    "<!-- gripper is appended at runtime by combine_arm_and_gripper_xml() -->",
    GRIPPER_FRAGMENT
)
src = src.replace(
    "<worldbody>",
    f"""<visual>
    <global offwidth="{W}" offheight="{H}"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 -0.01" size="1 1 0.01" rgba="0.3 0.3 0.32 1"/>
    <geom type="cylinder" fromto="0 0 0  0.2 0 0" size="0.005" rgba="1 0 0 1"/>
    <geom type="cylinder" fromto="0 0 0  0 0.2 0" size="0.005" rgba="0 1 0 1"/>
    <geom type="cylinder" fromto="0 0 0  0 0 0.2" size="0.005" rgba="0 0 1 1"/>
    <camera name="front" pos="0.0  1.0 0.4" xyaxes="-1 0 0   0 0 1"/>
    <camera name="side"  pos="1.0  0.0 0.4" xyaxes=" 0 1 0   0 0 1"/>
    <camera name="top"   pos="0.0  0.0 1.0" xyaxes=" 1 0 0   0 1 0"/>
    <camera name="iso"   pos="0.7 -0.7 0.6" xyaxes="0.707 0.707 0  -0.408 0.408 0.816"/>"""
)
out_xml = ROOT / "data" / "yam_inspect_with_gripper.xml"
out_xml.write_text(src)
print(f"wrote {out_xml}")

model = MjModel.from_xml_path(str(out_xml))
data = MjData(model)
renderer = Renderer(model, height=H, width=W)
print(f"model nq={model.nq} (6 arm + 2 gripper slides)")

poses = [
    ("zero",  np.zeros(6)),
    ("ready", np.array([0.0, 1.2, 1.2, 0.0, -0.3, 0.0])),
    ("up",    np.array([0.0, 1.5708, 0.0, 0.0, 0.0, 0.0])),
]
cams = ["front", "side", "top", "iso"]

out_dir = ROOT / "data" / "inspect"
for pose_label, q in poses:
    # arm joints are first 6
    for i, val in enumerate(q):
        data.qpos[i] = val
    mujoco.mj_forward(model, data)
    panels = []
    for cam in cams:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panels.append(bgr)
    top = np.hstack([panels[0], np.full((H, 8, 3), 255, np.uint8), panels[1]])
    bot = np.hstack([panels[2], np.full((H, 8, 3), 255, np.uint8), panels[3]])
    sep_h = np.full((8, top.shape[1], 3), 255, np.uint8)
    grid = np.vstack([top, sep_h, bot])
    cv2.putText(grid, f"q={list(q.round(2))}", (10, grid.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    out = out_dir / f"yam_inspect_grip_{pose_label}.png"
    cv2.imwrite(str(out), grid)
    print(f"wrote {out}")
