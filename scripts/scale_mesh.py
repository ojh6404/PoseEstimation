import os
import trimesh
import pyrender
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

def process_mesh(file_path, output_dir, scale_factor=1.0, vis=False):
    mesh = trimesh.load(file_path)
    original_bbox_size = get_bounding_box_size(mesh)
    if scale_factor != 1.0:
        mesh = scale_mesh(mesh, scale_factor)
        scaled_bbox_size = get_bounding_box_size(mesh)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        scaled_file_path = os.path.join(output_dir, os.path.basename(file_path))
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.export(scaled_file_path)

    if vis:
        visualize_mesh(mesh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to mesh file")
    parser.add_argument("--scale", type=float, default=1000.0, help="Scale factor")
    parser.add_argument("--vis", action="store_true", help="Visualize the mesh")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    process_mesh(args.file_path, args.output_dir, args.scale, args.vis)
