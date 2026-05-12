"""PnP solver with automatic arm1↔arm2 swap detection.

The user labels clicks as `arm1` / `arm2` consistently, but they don't
necessarily know which physical gripper corresponds to the Vive's
`arm1_pose_*` vs `arm2_pose_*` in the dataset. We try both assignments
(identity and swap) and keep the one with lower reprojection error.

Uses ONLY the `base` landmark (midpoint between the two fingers at the
base) — unambiguous, no L/R convention issues.
"""
import json
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent

W = H = 1024
FOV_PIN_DEG = 60.0
F_PIN = (W/2) / np.tan(np.deg2rad(FOV_PIN_DEG/2))
K_PIN = np.array([[F_PIN, 0, W/2], [0, F_PIN, H/2], [0, 0, 1]], dtype=np.float64)

# `base` = midpoint between the two tip body origins in gripper local frame
BASE_OFFSET = np.array([0.0, 0.0, -0.0546], dtype=np.float64)

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

def gather(clicks, meta, swap):
    """Return (obj, img) arrays for solvePnP under a given arm assignment.
    swap=False : click.arm==1 → Vive arm1, etc.
    swap=True  : click.arm==1 → Vive arm2, click.arm==2 → Vive arm1.
    """
    obj, img = [], []
    skipped = 0
    for c in clicks:
        if c["land"] != "base":
            skipped += 1; continue
        clicked_arm = c["arm"]
        vive_arm = (3 - clicked_arm) if swap else clicked_arm
        m = meta[c["frame"]]
        if m["state"] is None: skipped += 1; continue
        s = m["state"]
        if vive_arm == 1:
            pos = np.array(s[7:10]); q = s[10:14]
        else:
            pos = np.array(s[23:26]); q = s[26:30]
        # Skip degenerate / outlier poses
        if (abs(q[3]-1) < 1e-6 and abs(q[0]) < 1e-6 and abs(q[1]) < 1e-6 and abs(q[2]) < 1e-6) \
           or np.linalg.norm(pos) < 1e-6 or np.max(np.abs(pos)) > 1.5:
            skipped += 1; continue
        Pw = pos + quat_to_R(q) @ BASE_OFFSET
        obj.append(Pw); img.append([c["u"], c["v"]])
    return np.array(obj, dtype=np.float64), np.array(img, dtype=np.float64), skipped

def solve(obj, img):
    if len(obj) < 4:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K_PIN, distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=15.0, iterationsCount=500,
    )
    if not ok: return None
    obj_in = obj[inl[:,0]]; img_in = img[inl[:,0]]
    ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K_PIN, None, rvec, tvec,
                                   useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K_PIN, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - img_in)**2, axis=1))))
    return dict(rvec=rvec, tvec=tvec, n_in=len(inl), n_total=len(obj), rms=rms)

def main():
    meta   = json.loads((ROOT / "data" / "frames_meta.json").read_text())
    clicks = json.loads((ROOT / "data" / "calibration_clicks.json").read_text())["clicks"]
    print(f"total clicks: {len(clicks)}")
    base_clicks = [c for c in clicks if c["land"] == "base"]
    print(f"base clicks:  {len(base_clicks)}")

    # Try several click subsets to find the most-consistent calibration.
    # In this dataset arm1_pose_* and arm2_pose_* seem to live in *different*
    # frames (per-arm tracker origins), so calibrating from both jointly
    # produces inconsistent residuals. We also try "arm2 only" and "arm1 only"
    # to recover at least one good per-arm calibration.
    subsets = {
        "all (identity)":          ([1, 2], False),
        "all (swap arm1<->arm2)":  ([1, 2], True),
        "arm2 only (identity)":    ([2],    False),
        "arm1 only (identity)":    ([1],    False),
        "arm2 only (swap)":        ([2],    True),
        "arm1 only (swap)":        ([1],    True),
    }
    results = {}
    for label, (arms, swap) in subsets.items():
        filt = [c for c in clicks if c["arm"] in arms]
        obj, img, skipped = gather(filt, meta, swap)
        print(f"\n=== {label} ===")
        print(f"  usable correspondences: {len(obj)} (skipped {skipped})")
        r = solve(obj, img)
        if r is None:
            print("  PnP failed"); continue
        print(f"  RANSAC inliers: {r['n_in']}/{r['n_total']}, RMS: {r['rms']:.2f} px")
        results[label] = r

    if not results:
        raise SystemExit("no PnP solution")

    # ---- Per-Vive-arm best camera ----
    # A subset uses "Vive arm X" when (subset is "arm X only" AND not swap) OR
    # (subset is "arm Y only" AND swap). Pick whichever fit is best for each
    # Vive arm.
    def pick_best(*candidates):
        cands = [(label, results[label]) for label in candidates if label in results]
        if not cands: return None, None
        # Prefer higher inlier count, break ties by lower RMS
        label, r = max(cands, key=lambda lr: (lr[1]['n_in'], -lr[1]['rms']))
        return label, r
    a1_label, a1 = pick_best("arm1 only (identity)", "arm2 only (swap)")
    a2_label, a2 = pick_best("arm2 only (identity)", "arm1 only (swap)")
    print(f"\nbest Vive-arm1 cam: {a1_label}")
    print(f"best Vive-arm2 cam: {a2_label}")

    if a1 is not None and a2 is not None:
        R_a1, _ = cv2.Rodrigues(a1['rvec']); t_a1 = a1['tvec'].flatten()
        R_a2, _ = cv2.Rodrigues(a2['rvec']); t_a2 = a2['tvec'].flatten()
        # Same physical camera in two worlds:
        #   P_cam = R_a1 P_A1 + t_a1 = R_a2 P_A2 + t_a2
        # arm1_to_arm2: P_A2 = R_a2.T R_a1 P_A1 + R_a2.T (t_a1 - t_a2)
        R_a1_to_a2 = R_a2.T @ R_a1
        t_a1_to_a2 = R_a2.T @ (t_a1 - t_a2)
        print(f"arm1 -> arm2 transform: rot={np.degrees(np.arccos(max(-1,min(1,(np.trace(R_a1_to_a2)-1)/2)))):.1f} deg, t_norm={np.linalg.norm(t_a1_to_a2)*100:.1f} cm")
    else:
        R_a1_to_a2 = np.eye(3); t_a1_to_a2 = np.zeros(3)

    # ---- Pick the camera to save ----
    # Use whichever per-Vive-arm fit has more inliers; that's the one the
    # downstream render/composite stages will use directly.
    if a1 is not None and (a2 is None or a1['n_in'] > a2['n_in']):
        best_label = a1_label; r = a1; cal_arm = 1
    elif a2 is not None:
        best_label = a2_label; r = a2; cal_arm = 2
    else:
        # Fallback to a "all" assignment if neither per-arm worked
        def score(r):
            ratio = r['n_in'] / max(r['n_total'], 1)
            return (round(ratio, 2), -r['rms'])
        best_label = max(results, key=lambda k: score(results[k]))
        r = results[best_label]
        cal_arm = 2 if "arm2" in best_label else 1
    r = results[best_label]
    print(f"\n-> best assignment: {best_label} "
          f"(inliers {r['n_in']}/{r['n_total']}, RMS {r['rms']:.2f}px)")
    R, _ = cv2.Rodrigues(r['rvec'])
    cam_pos = -R.T @ r['tvec'].flatten()
    print(f"  cam pos world: {cam_pos}")

    out = {
        "K": K_PIN.tolist(),
        "rvec": r['rvec'].flatten().tolist(),
        "tvec": r['tvec'].flatten().tolist(),
        "R_world_to_cam": R.tolist(),
        "cam_pos_world": cam_pos.tolist(),
        "reproj_rms_px": r['rms'],
        "n_inliers": int(r['n_in']),
        "n_correspondences": int(r['n_total']),
        "image_size": [W, H],
        "fov_pinhole_deg": FOV_PIN_DEG,
        "assignment": best_label,
        "calibrated_arm": int(cal_arm),
        "R_arm1_to_arm2": R_a1_to_a2.tolist(),
        "t_arm1_to_arm2": t_a1_to_a2.tolist(),
        "note": "Per-arm PnP with auto-swap detection; transform from best per-arm cameras",
    }
    out_path = ROOT / "data" / "scene_cam_calibration.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
