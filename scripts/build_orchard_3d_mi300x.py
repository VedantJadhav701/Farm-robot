"""
Video-to-3D Real Orchard Reconstruction (MI300X GPU)
===================================================
Reconstructs a registered 3D colored point cloud (.ply) from Turkey orchard
keyframes using:
  1. Depth-Anything-V2-Metric-Outdoor-Large-hf (FP16 on MI300X)
  2. Flying-pixel depth-gradient & sky filtering
  3. 3D-2D SolvePnP RANSAC visual odometry across orchard corridor
  4. Ground-plane & alley-axis alignment (+X forward, +Y up, Z across alley)
  5. 3.5 cm voxel downsampling -> binary PLY ready for Three.js WebGL /sim
"""

import json
import struct
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Write compact little-endian binary PLY (x, y, z, r, g, b)."""
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rec = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    rec["x"] = xyz[:, 0].astype(np.float32)
    rec["y"] = xyz[:, 1].astype(np.float32)
    rec["z"] = xyz[:, 2].astype(np.float32)
    rec["red"] = np.clip(rgb[:, 0], 0, 255).astype(np.uint8)
    rec["green"] = np.clip(rgb[:, 1], 0, 255).astype(np.uint8)
    rec["blue"] = np.clip(rgb[:, 2], 0, 255).astype(np.uint8)

    with open(path, "wb") as f:
        f.write(header)
        rec.tofile(f)


def estimate_step_pose(prev_bgr, curr_bgr, prev_depth, K):
    """Estimate SE(3) step [R, t] moving points from curr cam -> prev cam."""
    g0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g0.shape

    pts0 = cv2.goodFeaturesToTrack(g0, maxCorners=600, qualityLevel=0.01, minDistance=12)
    if pts0 is None or len(pts0) < 25:
        return np.eye(3), np.array([0.0, 0.0, 0.28])

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, winSize=(21, 21), maxLevel=3)
    st = st.ravel().astype(bool)
    p0 = pts0.reshape(-1, 2)[st]
    p1 = pts1.reshape(-1, 2)[st]

    u0 = np.clip(np.round(p0[:, 0]).astype(int), 0, w - 1)
    v0 = np.clip(np.round(p0[:, 1]).astype(int), 0, h - 1)
    z0 = prev_depth[v0, u0]

    valid = (z0 >= 1.2) & (z0 <= 11.0) & (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    if np.sum(valid) < 20:
        return np.eye(3), np.array([0.0, 0.0, 0.28])

    p0_v = p0[valid]
    p1_v = p1[valid]
    z0_v = z0[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X0 = (p0_v[:, 0] - cx) * z0_v / fx
    Y0 = (p0_v[:, 1] - cy) * z0_v / fy
    pts3d = np.stack([X0, Y0, z0_v], axis=1).astype(np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, p1_v.astype(np.float64), K, None, reprojectionError=3.5, iterationsCount=200
    )
    if not ok or inliers is None or len(inliers) < 12:
        return np.eye(3), np.array([0.0, 0.0, 0.28])

    R_0_to_1, _ = cv2.Rodrigues(rvec)
    # Camera motion in frame 0 coords: T_1_in_0 = inv(T_0_to_1)
    R_1_to_0 = R_0_to_1.T
    t_1_in_0 = (-R_0_to_1.T @ tvec).ravel()

    # Constrain wild outliers (handheld orchard walk: primarily forward +Z, small lateral/pitch)
    step_norm = float(np.linalg.norm(t_1_in_0))
    if step_norm > 0.75 or t_1_in_0[2] < -0.05:
        return np.eye(3), np.array([0.0, 0.0, 0.28])

    # Dampen large roll/pitch angles for upright horizon
    sy = np.sqrt(R_1_to_0[0, 0] ** 2 + R_1_to_0[1, 0] ** 2)
    yaw = np.arctan2(R_1_to_0[0, 2], R_1_to_0[2, 2])
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return R_yaw, np.clip(t_1_in_0, [-0.15, -0.08, 0.05], [0.15, 0.08, 0.45])


def voxel_downsample_colored(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float = 0.035):
    """Fast vectorized 3D voxel averaging without external deps."""
    q = np.floor(xyz / voxel_size).astype(np.int64)
    # Shift to positive for unique hashing
    q -= q.min(axis=0, keepdims=True)
    stride_y = int(q[:, 2].max()) + 2
    stride_x = int(q[:, 1].max() + 2) * stride_y
    keys = q[:, 0] * stride_x + q[:, 1] * stride_y + q[:, 2]

    order = np.argsort(keys)
    keys_s = keys[order]
    xyz_s = xyz[order]
    rgb_s = rgb[order].astype(np.float32)

    uniq_mask = np.empty(len(keys_s), dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(uniq_mask)
    counts = np.diff(np.append(starts, len(keys_s)))

    xyz_avg = np.add.reduceat(xyz_s, starts, axis=0) / counts[:, None]
    rgb_avg = np.add.reduceat(rgb_s, starts, axis=0) / counts[:, None]
    return xyz_avg.astype(np.float32), np.clip(rgb_avg, 0, 255).astype(np.uint8)


def main():
    work = Path("/workspace/orchard3d")
    frames_dir = work / "frames"
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"Found {len(frame_paths)} keyframes.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    print(f"Loading {model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    images_bgr = [cv2.imread(str(p)) for p in frame_paths]
    h, w = images_bgr[0].shape[:2]
    fx = fy = float(w) * 0.95
    cx, cy = w * 0.5, h * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    depths = []
    with torch.no_grad():
        for i in range(0, len(images_bgr), 8):
            batch_pil = [
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                for img in images_bgr[i : i + 8]
            ]
            inputs = processor(images=batch_pil, return_tensors="pt").to(device)
            pred = model(**inputs).predicted_depth
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze(1)
            for d in pred.cpu().numpy():
                depths.append(d.astype(np.float32))
    print(f"Computed metric depth for {len(depths)} frames.")

    # Estimate chained 3D world poses T_world_cam
    poses = [np.eye(4, dtype=np.float64)]
    for t in range(1, len(images_bgr)):
        R_step, t_step = estimate_step_pose(images_bgr[t - 1], images_bgr[t], depths[t - 1], K)
        T_step = np.eye(4, dtype=np.float64)
        T_step[:3, :3] = R_step
        T_step[:3, 3] = t_step
        poses.append(poses[-1] @ T_step)

    # Precompute pixel grid
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ux = (uu - cx) / fx
    vy = (vv - cy) / fy

    all_xyz = []
    all_rgb = []
    ground_samples = []

    for idx, (bgr, d, T) in enumerate(zip(images_bgr, depths, poses)):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
        grad = gx + gy

        is_sky = (vv < h * 0.30) & (hsv[:, :, 1] < 38) & (hsv[:, :, 2] > 195)
        valid = (d >= 0.85) & (d <= 10.5) & (grad < 0.14 * d) & (~is_sky)

        # Subsample density cleanly (every 2nd pixel)
        valid[1::2, :] = False

        X_c = ux[valid] * d[valid]
        Y_c = vy[valid] * d[valid]
        Z_c = d[valid]
        pts_c = np.stack([X_c, Y_c, Z_c, np.ones_like(Z_c)], axis=1)
        pts_w = (T @ pts_c.T).T[:, :3]

        rgb = bgr[:, :, ::-1][valid]
        all_xyz.append(pts_w)
        all_rgb.append(rgb)

        # Collect alley-floor samples (bottom center of frame)
        alley_mask = valid & (vv > h * 0.72) & (uu > w * 0.32) & (uu < w * 0.68)
        if np.any(alley_mask):
            g_c = np.stack(
                [
                    ux[alley_mask] * d[alley_mask],
                    vy[alley_mask] * d[alley_mask],
                    d[alley_mask],
                    np.ones(np.sum(alley_mask)),
                ],
                axis=1,
            )
            ground_samples.append((T @ g_c.T).T[:, :3])

    xyz_cam_world = np.concatenate(all_xyz, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0)

    # Convert Optical World (X right, Y down, Z forward) -> WebGL Orchard (X=Z_fwd, Y=-Y_down, Z=X_right)
    pts_webgl = np.stack([xyz_cam_world[:, 2], -xyz_cam_world[:, 1], xyz_cam_world[:, 0]], axis=1)

    if ground_samples:
        gs = np.concatenate(ground_samples, axis=0)
        gs_webgl = np.stack([gs[:, 2], -gs[:, 1], gs[:, 0]], axis=1)
        ground_y = float(np.percentile(gs_webgl[:, 1], 25))
        alley_z = float(np.median(gs_webgl[:, 2]))
    else:
        ground_y = float(np.percentile(pts_webgl[:, 1], 10))
        alley_z = 0.0

    pts_webgl[:, 1] -= ground_y
    pts_webgl[:, 2] -= alley_z

    # Keep orchard corridor slice (-0.3m <= Y <= 3.5m height, -4.5m <= Z <= 4.5m alley width)
    keep = (pts_webgl[:, 1] >= -0.25) & (pts_webgl[:, 1] <= 3.6) & (np.abs(pts_webgl[:, 2]) <= 4.8)
    pts_webgl = pts_webgl[keep]
    rgb_all = rgb_all[keep]

    pts_vox, rgb_vox = voxel_downsample_colored(pts_webgl, rgb_all, voxel_size=0.032)
    if len(pts_vox) > 550000:
        step = max(1, len(pts_vox) // 550000)
        pts_vox = pts_vox[::step]
        rgb_vox = rgb_vox[::step]

    # Align start of reconstructed orchard row to X=0.0 so the rover drives directly into it
    pts_vox[:, 0] -= float(np.percentile(pts_vox[:, 0], 1))

    out_ply = work / "orchard_turkey_web.ply"
    write_binary_ply(out_ply, pts_vox, rgb_vox)

    meta = {
        "source": "orchard_turkey_480p.mp4 (118s-136s)",
        "keyframes": len(images_bgr),
        "vertices": int(len(pts_vox)),
        "ply_mb": round(out_ply.stat().st_size / (1024 * 1024), 2),
        "bounds_x": [round(float(pts_vox[:, 0].min()), 2), round(float(pts_vox[:, 0].max()), 2)],
        "bounds_y": [round(float(pts_vox[:, 1].min()), 2), round(float(pts_vox[:, 1].max()), 2)],
        "bounds_z": [round(float(pts_vox[:, 2].min()), 2), round(float(pts_vox[:, 2].max()), 2)],
    }
    (work / "orchard_3d_meta.json").write_text(json.dumps(meta, indent=2))
    print("3D_RECON_COMPLETE:", json.dumps(meta))


if __name__ == "__main__":
    main()
