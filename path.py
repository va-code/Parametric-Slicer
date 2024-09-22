import os
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import distance

# Define the output folder path
output_folder = "DecompositionOUTPUT"
adjacency_file = os.path.join(output_folder, "adjacency_list.txt")

# Load the adjacency list
if not os.path.exists(adjacency_file):
    print(f"Error: {adjacency_file} does not exist.")
    adjacency_list = []
else:
    adjacency_list = []
    with open(adjacency_file, "r") as f:
        for line in f:
            i, j = map(int, line.split())
            adjacency_list.append([i, j])

    print("Adjacency List:")
    print(adjacency_list)

# Create a dictionary to easily access the connections for each node
connections_dict = {}
for i, j in adjacency_list:
    if i not in connections_dict:
        connections_dict[i] = []
    if j not in connections_dict:
        connections_dict[j] = []
    connections_dict[i].append(j)
    connections_dict[j].append(i)


# Count the connections for each node
connection_count = {}
for i, j in adjacency_list:
    if i in connection_count:
        connection_count[i] += 1
    else:
        connection_count[i] = 1
    if j in connection_count:
        connection_count[j] += 1
    else:
        connection_count[j] = 1

# Sort the nodes based on the number of connections, descending
sorted_nodes = sorted(connection_count.items(), key=lambda x: x[1], reverse=True)
print("Sorted Nodes by Connection Count:")
print(sorted_nodes)


# Load the decomposed meshes
meshes = []
mesh_files = [f for f in os.listdir(output_folder) if f.startswith("mesh_") and f.endswith(".stl")]
mesh_files.sort()  # Ensure files are in the correct order

for mesh_file in mesh_files:
    mesh = trimesh.load(os.path.join(output_folder, mesh_file), force="mesh")
    #print(mesh)
    meshes.append(mesh)

# Calculate centers of the meshes
def calculate_centers(meshes):
    centers = []
    for mesh in meshes:
        centers.append(mesh.centroid)
    return centers


# Create the ordered list based on distance
def create_ordered_list_by_distance(connections_dict, centers, start_node):
    ordered_list = []
    visited = set()
    
    # Start with the first mesh
    #print(sorted_nodes[0][0])
    current_node = start_node
    ordered_list.append(current_node)
    visited.add(current_node)
    
    while len(ordered_list) < len(centers):
        current_center = centers[current_node]
        min_dist = float('inf')
        next_node = None
        
        for neighbor in connections_dict.get(current_node, []):
            if neighbor not in visited:
                dist = distance.euclidean(current_center, centers[neighbor])
                if dist < min_dist:
                    min_dist = dist
                    next_node = neighbor
        
        if next_node is not None:
            ordered_list.append(next_node)
            visited.add(next_node)
            current_node = next_node
        else:
            # If no unvisited neighbors, pick the next unvisited node
            for node in range(len(centers)):
                if node not in visited:
                    current_node = node
                    ordered_list.append(current_node)
                    visited.add(current_node)
                    break
        #print(f"Visited nodes: {visited}")
        #print(f"Current ordered list: {ordered_list}")
    
    return ordered_list

def create_ordered_list_by_closest_points(connections_dict, meshes, start_node):
    ordered_list = []
    visited = set()

    # Start with the first mesh
    current_node = start_node
    ordered_list.append(current_node)
    visited.add(current_node)

    while len(ordered_list) < len(meshes):
        min_dist = float('inf')
        next_node = None

        # Get all edges of the current mesh
        current_mesh = meshes[current_node]
        #print("this is the current mesh")
        #print(current_mesh)
        current_points = current_mesh.vertices

        for neighbor in connections_dict.get(current_node, []):
            if neighbor not in visited:
                neighbor_mesh = meshes[neighbor]
                neighbor_points = neighbor_mesh.vertices

                # Calculate minimum distance between points of current mesh and neighbor mesh
                dist_matrix = distance.cdist(current_points, neighbor_points, 'euclidean')
                closest_dist = dist_matrix.min()
                
                if closest_dist < min_dist:
                    min_dist = closest_dist
                    next_node = neighbor

        if next_node is not None:
            ordered_list.append(next_node)
            visited.add(next_node)
            current_node = next_node
        else:
            # If no unvisited neighbors, pick the next unvisited node
            for node in range(len(meshes)):
                if node not in visited:
                    current_node = node
                    ordered_list.append(current_node)
                    visited.add(current_node)
                    break

        #print(f"Visited nodes: {visited}")
        #print(f"Current ordered list: {ordered_list}")

    return ordered_list
    ordered_list = []
    visited = set()

    # Start with the first mesh
    current_node = sorted_nodes[0][0]
    ordered_list.append(current_node)
    visited.add(current_node)

    while len(ordered_list) < len(meshes):
        min_dist = float('inf')
        next_node = None

        # Get all edges of the current mesh
        current_mesh = meshes[current_node]
        current_points = current_mesh.vertices

        for neighbor in connections_dict.get(current_node, []):
            if neighbor not in visited:
                neighbor_mesh = meshes[neighbor]
                neighbor_points = neighbor_mesh.vertices

                # Calculate minimum distance between points of current mesh and neighbor mesh
                for p1 in current_points:
                    for p2 in neighbor_points:
                        dist = np.linalg.norm(p1 - p2)
                        if dist < min_dist:
                            min_dist = dist
                            next_node = neighbor

        if next_node is not None:
            ordered_list.append(next_node)
            visited.add(next_node)
            current_node = next_node
        else:
            # If no unvisited neighbors, pick the next unvisited node
            for node in range(len(meshes)):
                if node not in visited:
                    current_node = node
                    ordered_list.append(current_node)
                    visited.add(current_node)
                    break

        print(f"Visited nodes: {visited}")
        print(f"Current ordered list: {ordered_list}")

    return ordered_list

def create_ordered_list_by_convex_hull(connections_dict, meshes, start_node):
    ordered_list = []
    visited = set()

    # Start with the specified node
    current_node = start_node
    ordered_list.append(current_node)
    visited.add(current_node)

    while len(ordered_list) < len(meshes):
        min_hull_volume = float('inf')
        next_node = None

        # Get all points of the current convex hull
        current_points = np.vstack([meshes[node].vertices for node in ordered_list])
        current_hull = trimesh.convex.convex_hull(current_points)

        for node in ordered_list:
            for neighbor in connections_dict.get(node, []):
                if neighbor not in visited:
                    neighbor_points = meshes[neighbor].vertices
                    new_points = np.vstack([current_hull.vertices, neighbor_points])
                    new_hull = trimesh.convex.convex_hull(new_points)
                    hull_volume = new_hull.volume
                    
                    if hull_volume < min_hull_volume:
                        min_hull_volume = hull_volume
                        next_node = neighbor

        if next_node is not None:
            ordered_list.append(next_node)
            visited.add(next_node)
        else:
            # If no unvisited neighbors, pick the next unvisited node
            for node in range(len(meshes)):
                if node not in visited:
                    ordered_list.append(node)
                    visited.add(node)
                    break

        #print(f"Visited nodes: {visited}")
        #print(f"Current ordered list: {ordered_list}")

    return ordered_list

centers = calculate_centers(meshes)
start_node =sorted_nodes[0][0]
#ordered_list = create_ordered_list_by_distance(connections_dict, centers, sorted_nodes)
#ordered_list = create_ordered_list_by_closest_points(connections_dict, meshes, sorted_nodes)
ordered_list = create_ordered_list_by_convex_hull(connections_dict, meshes, start_node)

np.savetxt(os.path.join(output_folder,'ordered_list.txt'), ordered_list, fmt='%d', header='', comments='')
print("Ordered List of Nodes saved:")
print(ordered_list)

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.jet(np.linspace(0, 1, len(meshes)))  # Generate different colors for each mesh

current_frame = 0

def update_plot(frame, ordered_list, meshes, ax, view_init=None):
    ax.cla()
    displayed_nodes = set(ordered_list[:frame+1])
    
    for idx in ordered_list[:frame+1]:
        mesh = meshes[idx]
        faces = mesh.faces
        vertices = mesh.vertices
        poly3d = [[vertices[vert_id] for vert_id in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=colors[idx], linewidths=0.1, edgecolors='k', alpha=0.5))
    
    if frame < len(ordered_list):
        current_node = ordered_list[frame]
        connected_nodes = connections_dict.get(current_node, [])
        print(f"Adding node: {current_node}")
        for node in connected_nodes:
            status = "already displayed" if node in displayed_nodes else "not displayed yet"
            c = connections_dict[node]
            print(f"  - connected to node: {node} ({status}) Connections: {c}")
    
    if view_init:
        ax.view_init(elev=view_init[0], azim=view_init[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Scatter Plot of Mesh Vertices")

def on_key(event):
    global current_frame
    view_init = (ax.elev, ax.azim)
    if event.key == 'j':
        if current_frame < len(ordered_list) - 1:
            current_frame += 1
    elif event.key == 'k':
        if current_frame > 0:
            current_frame -= 1
    update_plot(current_frame, ordered_list, meshes, ax, view_init)
    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot(current_frame, ordered_list, meshes, ax)
plt.show()
