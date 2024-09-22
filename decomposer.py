import os
import CGAL
import coacd
import trimesh
import numpy as np
import CGAL.CGAL_Kernel as Kernel
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_file = "A.stl"
#input_file = "fractal.stl"
#input_file = "Stanford dragon.stl"
output_folder = "DecompositionOUTPUT"

mesh = trimesh.load(input_file, force="mesh")
mesh = coacd.Mesh(mesh.vertices, mesh.faces)
meshes = coacd.run_coacd(mesh)
'''
for part in meshes:
    print("decomp: ")
    print(part)
'''

def scale_vertices(vertices, scale_factor=1.0):
    centroid = np.mean(vertices, axis=0)
    scaled_vertices = centroid + scale_factor * (vertices - centroid)
    return scaled_vertices
''' 
def remove_duplicate_connections(connections):
    unique_connections = set()
    filtered_connections = []

    for conn in connections:
        # Sort the connection to ensure [1, 3] and [3, 1] are treated as the same
        sorted_conn = tuple(sorted(conn))
        if sorted_conn not in unique_connections:
            unique_connections.add(sorted_conn)
            filtered_connections.append(list(sorted_conn))

    return filtered_connections
    
def check_mesh_adjacency(mesh1, mesh2):
    # Extract vertices and faces from each mesh
    mesh1_vertices = mesh1[0]
    mesh1_faces = mesh1[1]
    mesh2_vertices = mesh2[0]
    mesh2_faces = mesh2[1]

    # Scale vertices out from their centroid by 1% for adjacency check
    scaled_mesh1_vertices = scale_vertices(mesh1_vertices)
    scaled_mesh2_vertices = scale_vertices(mesh2_vertices)

    # Ensure vertices and faces have correct data types
    scaled_mesh1_vertices = scaled_mesh1_vertices.astype(np.float64)
    mesh1_faces = mesh1_faces.astype(np.int32)
    scaled_mesh2_vertices = scaled_mesh2_vertices.astype(np.float64)
    mesh2_faces = mesh2_faces.astype(np.int32)

    # Convert face indices to CGAL triangles using scaled vertices
    mesh1_triangles = [Kernel.Triangle_3(
        Kernel.Point_3(*scaled_mesh1_vertices[face[0]]),
        Kernel.Point_3(*scaled_mesh1_vertices[face[1]]),
        Kernel.Point_3(*scaled_mesh1_vertices[face[2]]))
        for face in mesh1_faces]

    mesh2_triangles = [Kernel.Triangle_3(
        Kernel.Point_3(*scaled_mesh2_vertices[face[0]]),
        Kernel.Point_3(*scaled_mesh2_vertices[face[1]]),
        Kernel.Point_3(*scaled_mesh2_vertices[face[2]]))
        for face in mesh2_faces]

    # Check for intersections between triangles
    for tri1 in mesh1_triangles:
        for tri2 in mesh2_triangles:
            if CGAL.CGAL_Kernel.do_intersect(tri1, tri2):
                return True

    return False

# Example usage:
adjacency_list = []

# Check adjacency for each pair of meshes
for i in range(len(meshes)):
    for j in range(i + 1, len(meshes)):
        is_adjacent = check_mesh_adjacency(meshes[i], meshes[j])
        if is_adjacent:
            adjacency_list.append([i, j])

adjacency_list = remove_duplicate_connections(adjacency_list)
print("Meshes are adjacent:", adjacency_list)

with open(os.path.join(output_folder, "adjacency_list.txt"), "w") as f:
    for pair in adjacency_list:
        f.write(f"{pair[0]} {pair[1]}\n")
'''
for idx, part in enumerate(meshes):
    vertices, faces = part
    decomposed_mesh = trimesh.Trimesh(vertices, faces)
    mesh_filename = os.path.join(output_folder, f"mesh_{idx}.stl")
    decomposed_mesh.export(mesh_filename)
    print(f"Saved {mesh_filename}")
