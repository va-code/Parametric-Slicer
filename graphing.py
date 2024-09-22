import os
import trimesh
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the output folder path
output_folder = "DecompositionOUTPUT"

# Load the adjacency list from the text file
adjacency_list = []
with open(os.path.join(output_folder, "adjacency_list.txt"), "r") as f:
    for line in f:
        i, j = map(int, line.split())
        adjacency_list.append([i, j])

# Load the decomposed meshes
meshes = []
mesh_files = [f for f in os.listdir(output_folder) if f.startswith("mesh_") and f.endswith(".stl")]
mesh_files.sort()  # Ensure files are in the correct order

for mesh_file in mesh_files:
    mesh = trimesh.load(os.path.join(output_folder, mesh_file), force="mesh")
    meshes.append((mesh.vertices, mesh.faces))

# Create a graph
G = nx.Graph()

# Add edges from the adjacency list
for connection in adjacency_list:
    G.add_edge(connection[0], connection[1])

# Choose a layout algorithm
# pos = nx.spring_layout(G, iterations=200)  # or nx.kamada_kawai_layout(G)
pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai layout to minimize edge crossings

# Draw the graph
plt.figure(figsize=(10, 7))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=12, font_weight='bold', edge_color='gray')
plt.title("Mesh Adjacency Graph")
plt.show()

# Plotting the points of each mesh in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.jet(np.linspace(0, 1, len(meshes)))  # Generate different colors for each mesh

for i, (vertices, faces) in enumerate(meshes):
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=colors[i], label=f'Mesh {i+1}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.title("3D Scatter Plot of Mesh Vertices")
plt.show()
