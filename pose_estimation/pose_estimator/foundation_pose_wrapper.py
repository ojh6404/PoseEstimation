"""
Foundation Pose Wrapper
"""

from typing import List, Tuple, Optional
import sys
import numpy as np
from transformations import euler_matrix
import logging
import trimesh
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from pose_estimation import FOUNDATION_POSE_DIR
from pose_estimation.pose_estimator.wrapper_base import PoseEstimationWrapperBase
from pose_estimation.utils import (
    depth2xyzmap,
    pts2o3d,
    set_seed,
    draw_posed_3d_box,
    draw_xyz_axis,
    make_mesh_tensors,
    erode_depth,
    sample_views_icosphere,
    compute_mesh_diameter,
    bilateral_filter_depth,
    depth2xyzmap_batch,
)

# Add FoundationPose to the path
sys.path.insert(0, FOUNDATION_POSE_DIR)
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from datareader import YcbineoatReader
import mycpp.build.mycpp as mycpp

logging.getLogger().setLevel(logging.WARNING)


class FoundationPoseEstimator(PoseEstimationWrapperBase):
    """
    Pose Estimation Wrapper for FoundationPose
    """

    def __init__(
        self,
        mesh_file: str,
        mask: np.ndarray,
        K: np.ndarray,
        unit: str = "m",
        register_refine_iter: int = 5,
        track_refine_iter: int = 2,
        seed: int = 0,
        device: str = "cuda:0",
    ):
        """
        Initialize Pose Estimation Model

        Args:
            mesh_file (str): Path to mesh file
            mask (np.ndarray): np.bool mask of the object in first frame
            K (np.ndarray): Camera matrix, 3x3
            register_refine_iter (int): Register refine iteration for first pose estimation
            track_refine_iter (int): Track refine iteration for tracking
            seed (int): Random seed
            device (str): Device to use
        """
        super().__init__()

        self.mask = mask.astype(bool)
        self.K = K
        self.register_refine_iter = register_refine_iter
        self.track_refine_iter = track_refine_iter
        self.device = device
        self._initialize(
            mesh_file=mesh_file,
            seed=seed,
            unit=unit,
        )

    def _initialize(self, mesh_file: str, seed: int = 0, unit: str = "m") -> None:
        """
        Initialize Pose Estimation Model

        Args:
            mesh_file (str): Path to mesh file
            seed (int): Random seed
            unit (str): Unit of the mesh, default is "m"
        """

        set_seed(seed)

        mesh = trimesh.load(mesh_file)
        if unit == "mm":
            mesh.apply_scale(1e-3)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)

        self.ignore_normal_flip = True

        self.set_object(mesh=mesh, symmetry_tfs=None)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)

        self.glctx = dr.RasterizeCudaContext()
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()

        self.pose_last = None  # Used for tracking; per the centered mesh

    def set_object(self, mesh: trimesh.Trimesh, symmetry_tfs: Optional[List[np.ndarray]] = None) -> None:
        """
        Set object for pose estimation

        Args:
            mesh (trimesh.Trimesh): Mesh
            symmetry_tfs (Optional[List[np.ndarray]]): Symmetry transformations
        """

        self.mesh = mesh
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        model_pts = mesh.vertices
        model_normals = mesh.vertex_normals

        max_xyz = model_pts.max(axis=0)
        min_xyz = model_pts.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        self.diameter = compute_mesh_diameter(model_pts=model_pts, n_sample=10000)
        self.vox_size = max(self.diameter / 20.0, 0.003)
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg
        pcd = pts2o3d(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=self.device)
        self.normals = F.normalize(
            torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device=self.device), dim=-1
        )

        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device=self.device, dtype=torch.float)

    def get_tf_to_centered_mesh(self) -> torch.Tensor:
        """
        Get transformation to centered mesh

        Returns:
            torch.Tensor: Transformation to centered mesh
        """

        tf_to_center = torch.eye(4, dtype=torch.float, device=self.device)
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device=self.device, dtype=torch.float)
        return tf_to_center

    def make_rotation_grid(self, min_n_views: int = 40, inplane_step: int = 60) -> None:
        """
        Make rotation grid

        Args:
            min_n_views (int): Minimum number of views
            inplane_step (int): Inplane step
        """
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        self.rot_grid = torch.as_tensor(rot_grid, device=self.device, dtype=torch.float)

    def generate_random_pose_hypo(self, K: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """
        Generate random pose hypothesis

        Args:
            K (np.ndarray): Camera matrix
            depth (np.ndarray): Depth map
            mask (np.ndarray): Mask

        Returns:
            torch.Tensor: Random pose hypothesis
        """
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device=self.device, dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    def guess_translation(self, depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Guess translation

        Args:
            depth (np.ndarray): Depth map
            mask (np.ndarray): Mask
            K (np.ndarray): Camera matrix

        Returns:
            torch.Tensor: Guess translation
        """

        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        return center.reshape(3)

    def register(
        self,
        K: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        ob_mask: np.ndarray,
        ob_id: Optional[int] = None,
        iteration: int = 5,
    ) -> np.ndarray:
        """
        Register object

        Args:
            K (np.ndarray): Camera matrix
            rgb (np.ndarray): RGB image
            depth (np.ndarray): Depth map
            ob_mask (np.ndarray): Object mask
            ob_id (Optional[int]): Object ID
            iteration (int): Iteration

        Returns:
            np.ndarray: Registered pose
        """

        depth = erode_depth(depth, radius=2, device=self.device)
        depth = bilateral_filter_depth(depth, radius=2, device=self.device)

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        if valid.sum() < 4:
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=K, depth=depth, mask=ob_mask)
        poses = poses.data.cpu().numpy()
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device=self.device, dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device=self.device)

        xyz_map = depth2xyzmap(depth, K)
        poses, _ = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.data.cpu().numpy(),
            normal_map=normal_map,
            xyz_map=xyz_map,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            iteration=iteration,
            get_vis=False,
        )

        scores, _ = self.scorer.predict(
            mesh=self.mesh,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=poses.data.cpu().numpy(),
            normal_map=normal_map,
            mesh_tensors=self.mesh_tensors,
            glctx=self.glctx,
            mesh_diameter=self.diameter,
            get_vis=False,
        )

        ids = torch.as_tensor(scores).argsort(descending=True)
        scores = scores[ids]
        poses = poses[ids]

        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores

        return best_pose.data.cpu().numpy()

    def track(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, iteration: int = 2) -> np.ndarray:
        """
        Track Object

        Args:
            rgb (np.ndarray): RGB image
            depth (np.ndarray): Depth map
            K (np.ndarray): Camera matrix
            iteration (int): Iteration

        Returns:
            np.ndarray: Tracked pose
        """

        if self.pose_last is None:
            raise RuntimeError("Pose last is None")

        depth = torch.as_tensor(depth, device=self.device, dtype=torch.float)
        depth = erode_depth(depth, radius=2, device=self.device)
        depth = bilateral_filter_depth(depth, radius=2, device=self.device)

        xyz_map = depth2xyzmap_batch(
            depth[None], torch.as_tensor(K, dtype=torch.float, device=self.device)[None], zfar=np.inf
        )[0]

        pose, _ = self.refiner.predict(
            mesh=self.mesh,
            mesh_tensors=self.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(),
            normal_map=None,
            xyz_map=xyz_map,
            mesh_diameter=self.diameter,
            glctx=self.glctx,
            iteration=iteration,
            get_vis=False,
        )
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)

    def predict(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Pose

        Args:
            rgb (np.ndarray): RGB image
            depth (np.ndarray): Depth map

        Returns:
            Tuple[np.ndarray, np.ndarray]: Pose and visualization
        """

        if self.pose_last is None:
            pose = self.register(
                K=self.K, rgb=rgb, depth=depth, ob_mask=self.mask, iteration=self.register_refine_iter
            )
        else:
            pose = self.track(rgb=rgb, depth=depth, K=self.K, iteration=self.track_refine_iter)
        center_pose = pose @ np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(K=self.K, img=rgb, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(
            rgb,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )

        return center_pose, vis
