"""Render the canonical i2rt yam.xml side-by-side with my inlined yam_chain
to verify kinematic equivalence at known joint configs.

Loads:
  A. data/i2rt/i2rt/robot_models/arm/yam/yam.xml (canonical, unmodified)
  B. an inlined-chain version (the same one scripts/27 and /28 use)

Then renders both with q = [0, 0, 0, 0, 0, 0], q = ready, etc. The two should
look identical. If they differ, my inlined chain has drifted.
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import mujoco
from mujoco import MjModel, MjData, Renderer

ROOT = Path(__file__).resolve().parent.parent
W = H = 640
YAM_DIR = ROOT / "data" / "i2rt" / "i2rt" / "robot_models" / "arm" / "yam"

# Build a self-contained version of the canonical XML with a camera
canonical_src = (YAM_DIR / "yam.xml").read_text()
# Insert a camera + visual block.  The mesh dir is the assets folder.
abs_assets = (YAM_DIR / "assets").as_posix()
canonical_src = canonical_src.replace(
    '<compiler angle="radian" meshdir="assets"/>',
    f'<compiler angle="radian" meshdir="{abs_assets}"/>'
)
canonical_src = canonical_src.replace(
    "<worldbody>",
    f"""<visual>
    <global offwidth="{W}" offheight="{H}"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/>
  </visual>
  <worldbody>
    <camera name="iso" pos="0.8 -0.8 0.7" xyaxes="0.707 0.707 0 -0.5 0.5 0.707"/>"""
)
canonical_path = ROOT / "data" / "yam_canonical.xml"
canonical_path.write_text(canonical_src)
print(f"wrote {canonical_path}")

# Inlined-chain (same as scripts/27 and 28 use, after revert)
def inlined_xml():
    return f"""
<mujoco model="yam_inlined">
  <compiler angle="radian" meshdir="{abs_assets}"/>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/>
  </visual>
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
    <camera name="iso" pos="0.8 -0.8 0.7" xyaxes="0.707 0.707 0 -0.5 0.5 0.707"/>
    <geom class="yam_dark" pos="-0.0374966 -0.0464005 0.187501" quat="0.707105 0 0.707108 0" mesh="base"/>
    <body name="link1" pos="0 0 0.067" quat="0.707105 0 0 -0.707108">
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
      <joint name="j1" pos="0 0 0" axis="0 0 1" range="-2.61799 3.05433"/>
      <geom class="yam_link" pos="-0.0464 0.0374968 0.119501" quat="0.499998 0.5 0.5 -0.500002" mesh="link1"/>
      <body name="link2" pos="-0.0329 0.02 0.0455" quat="0.499998 0.5 -0.500002 -0.5">
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
        <joint name="j2" pos="0 0 0" axis="0 0 1" range="0 3.65"/>
        <geom class="yam_link" pos="-0.0174977 -0.0740001 -0.07925" quat="0.499998 0.5 0.500002 0.5" mesh="link2"/>
        <body name="link3" pos="0.264 4.08431e-07 -0.06375" quat="9.38184e-07 -0.707105 -0.707108 9.38187e-07">
          <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
          <joint name="j3" pos="0 0 0" axis="0 0 1" range="0 3.66519"/>
          <geom class="yam_link" pos="0.0740003 -0.281499 -0.0813" quat="9.38184e-07 9.38187e-07 -0.707108 -0.707105" mesh="link3"/>
          <body name="link4" pos="0.0600003 -0.244999 -0.00205" quat="1.32679e-06 0 0 -1">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <joint name="j4" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
            <geom class="yam_link" pos="-0.0138003 0.0364989 -0.0787882" quat="0.707105 0.707108 0 0" mesh="link4"/>
            <body name="link5" pos="-0.0403003 0.0703851 -0.0323887" quat="9.38184e-07 -0.707105 -9.38187e-07 -0.707108">
              <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
              <joint name="j5" pos="0 0 0" axis="0 0 1" range="-1.5708 1.5708"/>
              <geom class="yam_dark" pos="-0.0463995 0.0311519 0.0265" quat="0.499998 -0.5 -0.5 -0.500002" mesh="link5"/>
              <body name="link6" pos="0 0 0" quat="1 0 0 0">
                <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
                <joint name="j6" pos="0 0 0" axis="0 0 1" range="-2.0944 2.0944"/>
                <site name="ee" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

inlined_path = ROOT / "data" / "yam_inlined.xml"
inlined_path.write_text(inlined_xml())

def render(xml_path, q):
    model = MjModel.from_xml_path(str(xml_path))
    data = MjData(model)
    for i, val in enumerate(q):
        # joint i+1 is at qpos index i
        data.qpos[i] = val
    mujoco.mj_forward(model, data)
    renderer = Renderer(model, height=H, width=W)
    renderer.update_scene(data, camera="iso")
    rgb = renderer.render()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), data

configs = [
    ("all_zeros", np.zeros(6)),
    ("ready",     np.array([0.0, 1.2, 1.2, 0.0, -0.3, 0.0])),
    ("j4_pos_pi2", np.array([0.0, 1.2, 1.2,  1.5708, 0.0, 0.0])),
    ("j4_neg_pi2", np.array([0.0, 1.2, 1.2, -1.5708, 0.0, 0.0])),
    ("j5_pos_pi2", np.array([0.0, 1.2, 1.2,  0.0,  1.5708, 0.0])),
    ("j5_neg_pi2", np.array([0.0, 1.2, 1.2,  0.0, -1.5708, 0.0])),
]

out_dir = ROOT / "data" / "inspect"
for label, q in configs:
    ca, dca = render(canonical_path, q)
    inl, dinl = render(inlined_path, q)
    # FK check: where does the last body (link6) end up?
    nb_ca = dca.xpos[-1]
    nb_in = dinl.xpos[-1]
    fk_err_mm = np.linalg.norm(nb_ca - nb_in) * 1000
    print(f"  {label:>14}: canonical link6 xpos={nb_ca.round(4)}  "
          f"inlined link6 xpos={nb_in.round(4)}  fk_err={fk_err_mm:.2f}mm")
    sep = np.full((H, 8, 3), 255, np.uint8)
    side = np.hstack([ca, sep, inl])
    cv2.putText(side, "CANONICAL yam.xml", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(side, "MY INLINED chain", (ca.shape[1]+16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(side, label, (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.imwrite(str(out_dir / f"yam_sanity_{label}.png"), side)
    print(f"    wrote {out_dir/f'yam_sanity_{label}.png'}")
