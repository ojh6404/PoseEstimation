#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

from pose_estimation.pose_estimator import FoundationPoseEstimator
from datareader import YcbineoatReader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", type=str)
    parser.add_argument("--test_scene_dir", type=str)
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)
    K = reader.K  # 3X3
    mask = reader.get_mask(0).astype(bool)  # np.bool

    pose_estimator = FoundationPoseEstimator(
        mesh_file=args.mesh_file,
        mask=mask,
        K=K,
        register_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        device=args.device,
    )

    for i in range(len(reader.color_files)):
        color = reader.get_color(i)  # np.uint8, rgb
        depth = reader.get_depth(i)  # np.float64, in meter
        pose, vis = pose_estimator.predict(rgb=color, depth=depth)
        cv2.imshow("test", vis[..., ::-1])  # BGR to RGB
        cv2.waitKey(1)
