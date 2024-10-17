#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import cv2
import numpy as np

from pose_estimation.pose_estimator import MegaposeEstimator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--example-dir", type=str, default="/home/leus/prog/megapose6d/local_data/examples/barbecue-sauce/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    example_dir = args.example_dir
    rgb = cv2.cvtColor(cv2.imread(example_dir + "image_rgb.png"), cv2.COLOR_BGR2RGB)
    mesh_file = example_dir + "meshes/barbecue-sauce/hope_000002.ply"

    with open(example_dir + "inputs/object_data.json", "r") as f:
        detection_data = json.load(f)

    with open(example_dir + "camera_data.json", "r") as f:
        camera_data = json.load(f)

    pose_estimator = MegaposeEstimator(
        model_name=args.model,
        mesh_file=mesh_file,
        K=np.array(camera_data["K"]),
        device=args.device,
    )

    output = pose_estimator.predict(rgb, bbox=np.array(detection_data[0]["bbox_modal"]))
    print("output", output)
