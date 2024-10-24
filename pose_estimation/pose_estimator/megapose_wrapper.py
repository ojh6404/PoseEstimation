"""
Megapose Wrapper
"""

from typing import List, Tuple, Optional
import numpy as np
import logging

from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import ObjectData
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.utils.load_model import NAMED_MODELS, load_named_model

from pose_estimation.pose_estimator.wrapper_base import PoseEstimationWrapperBase

logging.getLogger().setLevel(logging.WARNING)


class MegaposeEstimator(PoseEstimationWrapperBase):
    """
    Pose Estimation Wrapper for Megapose
    """

    def __init__(
        self,
        model_name: str,
        mesh_file: str,
        K: np.ndarray,
        device: str = "cuda",
    ):
        """
        Initialize Pose Estimation Model

        Args:
            model_name (str): Model Name
            mesh_file (str): Path to mesh file, obj or ply format.
            K (np.ndarray): Camera matrix, 3x3
            seed (int): Random seed
            device (str): Device to use
        """
        super().__init__()

        self.K = K
        self.device = device

        self._initialize(
            model_name=model_name,
            mesh_file=mesh_file,
        )

    def _initialize(self, model_name: str, mesh_file: str) -> None:
        """
        Initialize Pose Estimation Model

        Args:
            mesh_file (str): Path to mesh file
            seed (int): Random seed
        """

        rigid_objects = [RigidObject(label="object", mesh_path=mesh_file, mesh_units="mm")]  # TODO
        self.object_dataset = RigidObjectDataset(rigid_objects)
        self.model_info = NAMED_MODELS[model_name]
        if "cuda" in self.device:
            self.pose_estimator = load_named_model(model_name, self.object_dataset).cuda()
        else:
            self.pose_estimator = load_named_model(model_name, self.object_dataset)

    def predict(self, rgb: np.ndarray, bbox: np.ndarray, depth: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict Pose

        Args:
            rgb (np.ndarray): RGB image
            bbox (np.ndarray): 2D Bounding Box in xyxy format, [x1, y1, x2, y2]
            depth (np.ndarray): Depth map

        Returns:
            np.ndarray: Pose in 4x4 matrix format
        """

        detection = {"label": "object", "bbox_modal": bbox}
        detection = [ObjectData.from_json(detection)]
        if "cuda" in self.device:
            detections = make_detections_from_object_data(detection).cuda()
            observation = ObservationTensor.from_numpy(rgb, depth, self.K).cuda()
        else:
            detections = make_detections_from_object_data(detection)
            observation = ObservationTensor.from_numpy(rgb, depth, self.K)

        output, _ = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **self.model_info["inference_parameters"]
        )

        return output.poses.squeeze().cpu().numpy()
