import os
import coacd
import trimesh
import numpy as np
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
for idx, part in enumerate(meshes):
    vertices, faces = part
    decomposed_mesh = trimesh.Trimesh(vertices, faces)
    mesh_filename = os.path.join(output_folder, f"mesh_{idx}.stl")
    decomposed_mesh.export(mesh_filename)
    print(f"Saved {mesh_filename}")
