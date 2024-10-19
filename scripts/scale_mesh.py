import trimesh
import pyrender
import numpy as np
import argparse

def visualize_mesh(mesh):
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)

def get_bounding_box_size(mesh):
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    return size

def scale_mesh(mesh, scale_factor):
    mesh.apply_scale(scale_factor)
    return mesh

def process_mesh(file_path, scale_factor=1.0):
    mesh = trimesh.load(file_path)
    original_bbox_size = get_bounding_box_size(mesh)
    if scale_factor != 1.0:
        mesh = scale_mesh(mesh, scale_factor)
        scaled_bbox_size = get_bounding_box_size(mesh)
        scaled_file_path = file_path.rsplit('.', 1)[0] + f'_scaled_{scale_factor}.' + file_path.rsplit('.', 1)[1]
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.export(scaled_file_path)

    # 메쉬 시각화
    visualize_mesh(mesh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to mesh file")
    parser.add_argument("--scale_factor", type=float, default=1000.0, help="Scale factor")
    args = parser.parse_args()

    process_mesh(args.file_path, args.scale_factor)
