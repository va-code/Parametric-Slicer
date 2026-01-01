import os
import trimesh
import numpy as np
from scipy.spatial import KDTree
from profiler import profile, ProfilerManager, profile_block

# Initialize profiler from environment
ProfilerManager.set_debug_mode(os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))

# Define the output folder path
output_folder = "DecompositionOUTPUT"
output_adjacency_file = os.path.join(output_folder, "adjacency_list.txt")

# Ensure the output folder exists
if not os.path.exists(output_folder):
    print(f"Error: {output_folder} does not exist.")
    exit()

# Load the decomposed meshes
mesh_files = [f for f in os.listdir(output_folder) if f.startswith("mesh_") and f.endswith(".stl")]
mesh_files.sort()  # Ensure files are in the correct order

meshes = []
with profile_block("load_meshes"):
    for mesh_file in mesh_files:
        mesh = trimesh.load(os.path.join(output_folder, mesh_file), force="mesh")
        meshes.append(mesh)

@profile
def remove_duplicate_connections(connections):
    unique_connections = set()
    filtered_connections = []

    for conn in connections:
        sorted_conn = tuple(sorted(conn))
        if sorted_conn not in unique_connections:
            unique_connections.add(sorted_conn)
            filtered_connections.append(list(sorted_conn))

    return filtered_connections

@profile
def check_mesh_adjacency(mesh1, mesh2, threshold=1e-3):
    mesh1_kdtree = KDTree(mesh1.vertices)
    mesh2_kdtree = KDTree(mesh2.vertices)
    
    for vertex in mesh1.vertices:
        distance, _ = mesh2_kdtree.query(vertex)
        if distance < threshold:
            return True
    
    for vertex in mesh2.vertices:
        distance, _ = mesh1_kdtree.query(vertex)
        if distance < threshold:
            return True

    return False

# Check adjacency for each pair of meshes
with profile_block("compute_adjacency"):
    adjacency_list = []
    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            is_adjacent = check_mesh_adjacency(meshes[i], meshes[j])
            if is_adjacent:
                adjacency_list.append([i, j])

adjacency_list = remove_duplicate_connections(adjacency_list)
print("Meshes are adjacent:", adjacency_list)

# Save adjacency list to file
with profile_block("save_adjacency_list"):
    with open(output_adjacency_file, "w") as f:
        for pair in adjacency_list:
            f.write(f"{pair[0]} {pair[1]}\n")

print(f"Adjacency list saved to {output_adjacency_file}")

# Generate profiling report if debug mode is enabled
if ProfilerManager.is_debug_mode():
    profiling_folder = os.path.join(output_folder, "Profiling")
    os.makedirs(profiling_folder, exist_ok=True)
    ProfilerManager.print_report(os.path.join(profiling_folder, "Profiling_Adjaceny.txt"))
    ProfilerManager.save_csv_report(os.path.join(profiling_folder, "Profiling_Adjaceny.csv"))
