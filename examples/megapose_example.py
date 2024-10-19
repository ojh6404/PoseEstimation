#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pose_estimation.pose_estimator import MegaposeEstimator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--image", type=str, default="test/bottle/bottle.png")
    parser.add_argument(
        "--mesh",
        type=str,
        default="/home/leus/prog/PoseEstimation/test/bottle/Scaniverse_2024_10_19_140711_scaled_1000.0.obj",
    ) # should be absolute path
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    bbox = np.array([348, 196, 419, 365])
    K = np.array([[570.34, 0.0, 319.5], [0.0, 570.34, 239.5], [0.0, 0.0, 1.0]])

    pose_estimator = MegaposeEstimator(
        model_name=args.model,
        mesh_file=args.mesh,
        K=K,
        device=args.device,
    )

    output = pose_estimator.predict(rgb, bbox=bbox)  # [4x4 matrix]
    r = R.from_matrix(output[:3, :3])
    trans = output[:3, 3]
    quat = r.as_quat()

    print("Translation :", trans, "Quat :", quat)
