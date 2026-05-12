"""Debug IK convergence on a single frame."""
import json, numpy as np, cv2
import mujoco
from mujoco import MjModel, MjData
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Run the same XML build as script 23 by importing or copy the code...
# Simpler: just exec script 23's setup up to the loop.

# Use the same scene XML that script 23 wrote
xml_path = ROOT / "data" / "yam_full_calibrated.xml"
model = MjModel.from_xml_path(str(xml_path))
data = MjData(model)

y1_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"y1_j{j}") for j in range(1,7)]
y1_qadr = [model.jnt_qposadr[j] for j in y1_jnt_ids]
y1_dofadr = [model.jnt_dofadr[j] for j in y1_jnt_ids]
y1_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "y1_ee")
print(f"y1_jnt_ids: {y1_jnt_ids}")
print(f"y1_qadr:    {y1_qadr}")
print(f"y1_dofadr:  {y1_dofadr}")
print(f"y1_ee_site: {y1_ee_site}")
print(f"model.nq:   {model.nq}  model.nv: {model.nv}")
print(f"jnt_ranges:")
for j in y1_jnt_ids:
    print(f"  joint {j}: range {model.jnt_range[j]}")

# Seed pose
ready = np.array([0.0, 1.0, 1.2, 0.0, -0.3, 0.0])
for i, a in enumerate(y1_qadr):
    data.qpos[a] = ready[i]
mujoco.mj_forward(model, data)
ee0 = data.site_xpos[y1_ee_site].copy()
print(f"\nat ready pose, y1_ee position: {ee0}")

# Sample target: actual gripper position at frame 500 in cal frame
import json
meta = json.load(open(ROOT / "data" / "frames_meta.json"))
calib = json.load(open(ROOT / "data" / "scene_cam_calibration.json"))
CAL = calib['calibrated_arm']
R_a1_to_a2 = np.array(calib['R_arm1_to_arm2'])
t_a1_to_a2 = np.array(calib['t_arm1_to_arm2'])
R_a2_to_a1 = R_a1_to_a2.T
t_a2_to_a1 = -R_a2_to_a1 @ t_a1_to_a2

s = meta[500]['state']
vive_arm = 1
if vive_arm == 1:
    pos = np.array(s[7:10])
else:
    pos = np.array(s[23:26])
if vive_arm != CAL:
    if CAL == 1: pos = R_a2_to_a1 @ pos + t_a2_to_a1
    else: pos = R_a1_to_a2 @ pos + t_a1_to_a2
target = pos
print(f"target (frame 500 arm{vive_arm}): {target}")
print(f"distance from base ({-0.43, -0.10, -0.22}) to target: {np.linalg.norm(target - np.array([-0.43,-0.10,-0.22])):.3f}m")

# Single-step position IK iteration with logging
Jp = np.zeros((3, model.nv))
Jr = np.zeros((3, model.nv))
for it in range(50):
    mujoco.mj_forward(model, data)
    cur = data.site_xpos[y1_ee_site].copy()
    err = target - cur
    e = np.linalg.norm(err)
    if it < 5 or it % 10 == 0:
        print(f"  it {it}: err={e*1000:.2f}mm, cur={cur}")
    if e < 1e-4: break
    mujoco.mj_jacSite(model, data, Jp, Jr, y1_ee_site)
    J = Jp[:, y1_dofadr]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
    for i, a in enumerate(y1_qadr):
        data.qpos[a] += dq[i]
    for i, jid in enumerate(y1_jnt_ids):
        lo, hi = model.jnt_range[jid]
        data.qpos[y1_qadr[i]] = np.clip(data.qpos[y1_qadr[i]], lo, hi)

mujoco.mj_forward(model, data)
print(f"\nfinal y1_ee: {data.site_xpos[y1_ee_site]}")
print(f"final qpos: {[data.qpos[a] for a in y1_qadr]}")
