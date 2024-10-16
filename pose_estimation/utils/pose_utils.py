"""
Utility functions for 6D pose estimation.
"""

from typing import Optional
import numpy as np
import torch
import open3d as o3d
import cv2
import random
import trimesh
import scipy

import warp as wp

wp.init()


def depth2xyzmap(depth: np.ndarray, K: np.ndarray, uvs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert depth map to xyz map.
    Args:
        depth: (H,W) depth map
        K: (3,3) intrinsic matrix
        uvs: (N,2) pixel coordinates
    Returns:
        xyz_map: (H,W,3) xyz map
    """

    invalid_mask = depth < 0.001
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij")
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def pts2o3d(
    points: np.ndarray, colors: Optional[np.ndarray] = None, normals: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """
    Convert points to open3d point cloud.
    Args:
        points: (N,3) points
        colors: (N,3) colors
        normals: (N,3) normals
    Returns:
        cloud: open3d point cloud
    """

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def set_seed(random_seed: int) -> None:
    """
    Set random seed for reproducibility.
    Args:
        random_seed: random seed
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def draw_posed_3d_box(K, img, ob_in_cam, bbox, line_color=(0, 255, 0), linewidth=2):
    """Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    """
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (ob_in_cam @ to_homo(pts).T).T[:, :3]  # (2,3)
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(int)  # (2,2)
        img = cv2.line(
            img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA
        )
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img


def to_homo(pts: np.ndarray) -> np.ndarray:
    """ """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def draw_xyz_axis(
    color: np.ndarray,
    ob_in_cam: np.ndarray,
    scale: float = 0.1,
    K: np.ndarray = np.eye(3),
    thickness: int = 3,
    transparency: float = 0,
    is_input_rgb: bool = False,
) -> np.ndarray:
    if is_input_rgb:
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, xx, color=(0, 0, 255), thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, yy, color=(0, 255, 0), thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, zz, color=(255, 0, 0), thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    return tmp


def project_3d_to_2d(pt: np.ndarray, K: np.ndarray, ob_in_cam: np.ndarray) -> np.ndarray:
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


@wp.kernel(enable_backward=False)
def erode_depth_kernel(
    depth: wp.array(dtype=float, ndim=2),
    out: wp.array(dtype=float, ndim=2),
    radius: int,
    depth_diff_thres: float,
    ratio_thres: float,
    zfar: float,
):
    h, w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w >= W or h >= H:
        return
    d_ori = depth[h, w]
    if d_ori < 0.001 or d_ori >= zfar:
        out[h, w] = 0.0
    bad_cnt = float(0)
    total = float(0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            total += 1.0
            if cur_depth < 0.001 or cur_depth >= zfar or abs(cur_depth - d_ori) > depth_diff_thres:
                bad_cnt += 1.0
    if bad_cnt / total > ratio_thres:
        out[h, w] = 0.0
    else:
        out[h, w] = d_ori


def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device="cuda"):
    depth_wp = wp.from_torch(torch.as_tensor(depth, dtype=torch.float, device=device))
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(
        kernel=erode_depth_kernel,
        device=device,
        dim=[depth.shape[0], depth.shape[1]],
        inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar],
    )
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
        depth_out = depth_out.data.cpu().numpy()
    return depth_out


def sample_views_icosphere(n_views, subdivisions=None, radius=1):
    if subdivisions is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    else:
        subdivision = 1
        while True:
            mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            if mesh.vertices.shape[0] >= n_views:
                break
            subdivision += 1
    cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices), 1, 1))
    cam_in_obs[:, :3, 3] = mesh.vertices
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]  # (N,3)
    z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1, 1)
    x_axis = np.cross(up.reshape(1, 3), z_axis)
    invalid = (x_axis == 0).all(axis=-1)
    x_axis[invalid] = [1, 0, 0]
    x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1, 1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1, 1)
    cam_in_obs[:, :3, 0] = x_axis
    cam_in_obs[:, :3, 1] = y_axis
    cam_in_obs[:, :3, 2] = z_axis
    return cam_in_obs


def make_mesh_tensors(mesh, device="cuda", max_tex_size=None):
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))
        img = img[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors["tex"] = torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        mesh_tensors["uv_idx"] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            mesh.visual.vertex_colors = np.tile(
                np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1)
            )
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float) / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int),
            "vnormals": torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
        }
    )
    return mesh_tensors


def compute_mesh_diameter(model_pts=None, mesh=None, n_sample=1000):
    if mesh is not None:
        u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
        pts = u @ s
        diameter = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return float(diameter)

    if n_sample is None:
        pts = model_pts
    else:
        ids = np.random.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)
        pts = model_pts[ids]
    dists = np.linalg.norm(pts[None] - pts[:, None], axis=-1)
    diameter = dists.max()
    return diameter


@wp.kernel(enable_backward=False)
def bilateral_filter_depth_kernel(
    depth: wp.array(dtype=float, ndim=2),
    out: wp.array(dtype=float, ndim=2),
    radius: int,
    zfar: float,
    sigmaD: float,
    sigmaR: float,
):
    h, w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w >= W or h >= H:
        return
    out[h, w] = 0.0
    mean_depth = float(0.0)
    num_valid = int(0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            if cur_depth >= 0.001 and cur_depth < zfar:
                num_valid += 1
                mean_depth += cur_depth
    if num_valid == 0:
        return
    mean_depth /= float(num_valid)

    depthCenter = depth[h, w]
    sum_weight = float(0.0)
    sum = float(0.0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            if cur_depth >= 0.001 and cur_depth < zfar and abs(cur_depth - mean_depth) < 0.01:
                weight = wp.exp(
                    -float((u - w) * (u - w) + (h - v) * (h - v)) / (2.0 * sigmaD * sigmaD)
                    - (depthCenter - cur_depth) * (depthCenter - cur_depth) / (2.0 * sigmaR * sigmaR)
                )
                sum_weight += weight
                sum += weight * cur_depth
    if sum_weight > 0 and num_valid > 0:
        out[h, w] = sum / sum_weight


def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device="cuda"):
    if isinstance(depth, np.ndarray):
        depth_wp = wp.array(depth, dtype=float, device=device)
    else:
        depth_wp = wp.from_torch(depth)
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(
        kernel=bilateral_filter_depth_kernel,
        device=device,
        dim=[depth.shape[0], depth.shape[1]],
        inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR],
    )
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
        depth_out = depth_out.data.cpu().numpy()
    return depth_out


def depth2xyzmap_batch(depths, Ks, zfar):
    """
    @depths: torch tensor (B,H,W)
    @Ks: torch tensor (B,3,3)
    """
    bs = depths.shape[0]
    invalid_mask = (depths < 0.001) | (depths > zfar)
    H, W = depths.shape[-2:]
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
    vs = vs.reshape(-1).float().cuda()[None].expand(bs, -1)
    us = us.reshape(-1).float().cuda()[None].expand(bs, -1)
    zs = depths.reshape(bs, -1)
    Ks = Ks[:, None].expand(bs, zs.shape[-1], 3, 3)
    xs = (us - Ks[..., 0, 2]) * zs / Ks[..., 0, 0]  # (B,N)
    ys = (vs - Ks[..., 1, 2]) * zs / Ks[..., 1, 1]
    pts = torch.stack([xs, ys, zs], dim=-1)  # (B,N,3)
    xyz_maps = pts.reshape(bs, H, W, 3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps
