import os
import coacd
import trimesh
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from profiler import profile, ProfilerManager, profile_block

# Initialize profiler from environment
ProfilerManager.set_debug_mode(os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))

# Get input file from environment variable or use default
input_file = os.environ.get('INPUT_FILE', 'A.stl')
print(f"Processing input file: {input_file}")

# Alternative options (for manual runs):
#input_file = "fractal.stl"
#input_file = "Stanford dragon.stl"
output_folder = "DecompositionOUTPUT"

with profile_block("load_mesh"):
    mesh = trimesh.load(input_file, force="mesh")

with profile_block("convert_to_coacd_mesh"):
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)

with profile_block("run_coacd_decomposition"):
    # Run coacd decomposition with parameters that prevent mesh simplification
    # decimate=False ensures the mesh is not simplified, only decomposed
    meshes = coacd.run_coacd(
        mesh,
        threshold=0.05,        # Default decomposition threshold
        resolution=2000,       # Default resolution
        max_convex_hull=-1,    # No limit on convex hulls
        preprocess_mode="auto", # Automatic preprocessing
        preprocess_resolution=30, # Default preprocessing resolution
        pca=False,             # Don't use PCA preprocessing
        merge=True,            # Merge small parts (doesn't affect simplification)
        decimate=False         # CRITICAL: Don't simplify/decimate the mesh
    )

'''
for part in meshes:
    print("decomp: ")
    print(part)
'''

@profile
def scale_vertices(vertices, scale_factor=1.0):
    centroid = np.mean(vertices, axis=0)
    scaled_vertices = centroid + scale_factor * (vertices - centroid)
    return scaled_vertices

# Export each decomposed mesh
with profile_block("export_decomposed_meshes"):
    for idx, part in enumerate(meshes):
        with profile_block(f"export_mesh_{idx}"):
            vertices, faces = part
            decomposed_mesh = trimesh.Trimesh(vertices, faces)
            mesh_filename = os.path.join(output_folder, f"mesh_{idx}.stl")
            decomposed_mesh.export(mesh_filename)
            print(f"Saved {mesh_filename}")

# Generate profiling report if debug mode is enabled
if ProfilerManager.is_debug_mode():
    profiling_folder = os.path.join(output_folder, "Profiling")
    os.makedirs(profiling_folder, exist_ok=True)
    ProfilerManager.print_report(os.path.join(profiling_folder, "Profiling_Decomposer.txt"))
    ProfilerManager.save_csv_report(os.path.join(profiling_folder, "Profiling_Decomposer.csv"))
