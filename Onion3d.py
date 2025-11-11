import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from profiler import profile, ProfilerManager, profile_block

# Initialize profiler from environment
ProfilerManager.set_debug_mode(os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))

# Define the output folder path
output_folder = "DecompositionOUTPUT"

# Ensure the output folder exists
if not os.path.exists(output_folder):
    print(f"Error: {output_folder} does not exist.")
    exit()

# Load the decomposed meshes
mesh_files = [f for f in os.listdir(output_folder) if f.startswith("mesh_") and f.endswith(".stl")]
mesh_files.sort()  # Ensure files are in the correct order
layer_height = 0.1
meshes = []
for mesh_file in mesh_files:
    mesh = trimesh.load(os.path.join(output_folder, mesh_file), force='mesh')
    meshes.append(mesh)

Test_mesh = meshes[0]


@profile
def create_planes(mesh, direction_vector, layer_height):
    centroid = mesh.centroid
    planes = []

    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Create planes in the positive direction
    plane_offset = 0
    while True:
        plane_origin = centroid + plane_offset * direction_vector
        if mesh.bounds[0][2] <= plane_origin[2] <= mesh.bounds[1][2]:
            planes.append({'origin': plane_origin, 'normal': direction_vector})
            plane_offset += layer_height
        else:
            break

    # Create planes in the negative direction
    plane_offset = -layer_height
    while True:
        plane_origin = centroid + plane_offset * direction_vector
        if mesh.bounds[0][2] <= plane_origin[2] <= mesh.bounds[1][2]:
            planes.append({'origin': plane_origin, 'normal': direction_vector})
            plane_offset -= layer_height
        else:
            break

    return planes

@profile
def calculate_intersection_lines(mesh, planes):
    intersection_lines = []
    epsilon = 1e-6  # Small value to avoid division by zero
    
    for plane in planes:
        intersections = trimesh.intersections.mesh_plane(mesh, plane['normal'], plane['origin'])
        if intersections is not None and len(intersections) > 0:
            oriented_lines = []
            for line in intersections:
                for i in range(len(line) - 1):
                    start_point = line[i]
                    end_point = line[i + 1]
                    
                    # Calculate line direction
                    line_direction = end_point - start_point
                    line_dir_norm = np.linalg.norm(line_direction)
                    
                    # Skip degenerate lines (start == end)
                    if line_dir_norm < epsilon:
                        continue
                    
                    line_direction /= line_dir_norm
                    
                    # Calculate normal vector (perpendicular to both plane normal and line direction)
                    normal_vector = np.cross(plane['normal'], line_direction)
                    normal_vec_norm = np.linalg.norm(normal_vector)
                    
                    # Skip if cross product is zero (line is parallel to plane normal)
                    if normal_vec_norm < epsilon:
                        # Use a default perpendicular vector instead
                        # Find any vector perpendicular to line_direction
                        if abs(line_direction[0]) < 0.9:
                            normal_vector = np.cross(line_direction, [1, 0, 0])
                        else:
                            normal_vector = np.cross(line_direction, [0, 1, 0])
                        normal_vec_norm = np.linalg.norm(normal_vector)
                    
                    normal_vector /= normal_vec_norm
                    
                    # Sanity check: ensure normalized vector has magnitude ~1
                    magnitude_squared = np.dot(normal_vector, normal_vector)
                    if magnitude_squared > 1.05 or magnitude_squared < 0.95:
                        print(f"Warning: Normal vector magnitude = {np.sqrt(magnitude_squared)}")
                    
                    oriented_lines.append(np.concatenate([start_point, normal_vector]))
                    oriented_lines.append(np.concatenate([end_point, normal_vector]))
            
            if len(oriented_lines) > 0:
                intersection_lines.append(np.array(oriented_lines))

    return intersection_lines
    
# Initialize the Base Vector in a random direction
base_vector = np.random.rand(3) - 0.5
base_vector /= np.linalg.norm(base_vector)  # Normalize the vector

# Function to show lines
@profile
def show_lines(Mesh, GRAPH):
    # Store all primary and secondary intersection lines
    all_intersection_lines = []
    
    # Create a random 3D vector
    random_vector = np.random.rand(3) - 0.5
    random_vector /= np.linalg.norm(random_vector)  # Normalize the vector
    #print(f"Random direction vector: {random_vector}")

    # Create primary planes and calculate intersections
    primary_planes = create_planes(Mesh, random_vector, layer_height)
    #print(f"Number of primary planes created: {len(primary_planes)}")
    primary_intersection_lines = calculate_intersection_lines(Mesh, primary_planes)
    #print(f"Number of primary intersection lines: {len(primary_intersection_lines)}")
    all_intersection_lines.append(primary_intersection_lines)

    # Create secondary planes (90 degrees offset) and calculate intersections
    orthogonal_vector = np.cross(random_vector, [1, 0, 0])
    if np.linalg.norm(orthogonal_vector) == 0:  # Handle the case where the random vector is parallel to [1, 0, 0]
        orthogonal_vector = np.cross(random_vector, [0, 1, 0])
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)

    secondary_planes = create_planes(Mesh, orthogonal_vector, layer_height)
    #print(f"Number of secondary planes created: {len(secondary_planes)}")
    secondary_intersection_lines = calculate_intersection_lines(Mesh, secondary_planes)
    #print(f"Number of secondary intersection lines: {len(secondary_intersection_lines)}")
    all_intersection_lines.append(secondary_intersection_lines)
    if GRAPH:
        # Visualization of the intersection lines (commented out)
        fig = plt.figure(facecolor="black")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type("ortho")
        ax.plot_trisurf(Mesh.vertices[:, 0], Mesh.vertices[:, 1], Mesh.vertices[:, 2], triangles=Mesh.faces, color=(0, 0, 1, 0.5), linewidth=0.2, edgecolor='k')
        # Plot the primary intersection lines
        for lines in primary_intersection_lines:
            #print(lines)
            for line in lines:
                line = np.array(line)  # Convert to NumPy array if not already
                if len(line.shape) == 1:  # Check if it's 1D
                    #print(lines)
                    line = line.reshape(-1, 3)[0]  # Reshape to (n, 3) if necessary
                    #print(lines)
              
                #exit()
                ax.plot(lines[:, 0], lines[:, 1], lines[:, 2], 'r-')

        # Plot the secondary intersection lines
        for lines in secondary_intersection_lines:
            for line in lines:
                line = np.array(line)  # Convert to NumPy array if not already
                if len(line.shape) == 1:  # Check if it's 1D
                    line = line.reshape(-1, 3)[0]  # Reshape to (n, 3) if necessary
                ax.plot(lines[:, 0], lines[:, 1], lines[:, 2], 'g-')
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Intersection lines with planes")
        plt.show()

    return all_intersection_lines
    
@profile
def ensure_faces_outward(mesh):
    # Compute the centroid of the mesh
    centroid = mesh.centroid

    # Ensure that the mesh has face normals computed
    face_normals = mesh.face_normals

    # Iterate through each face and check if it points outward
    for i, face in enumerate(mesh.faces):
        # Get the vertices of the face
        vertices = mesh.vertices[face]
        
        # Compute the face normal
        face_normal = face_normals[i]
        
        # Compute the vector from the centroid to one of the vertices of the face
        centroid_to_face_vector = vertices[0] - centroid
        
        # Check if the normal is pointing outwards by computing the dot product
        dot_product = np.dot(face_normal, centroid_to_face_vector)
        
        if dot_product < 0:
            # If the dot product is negative, the face is pointing inwards
            # Flip the face normal by reversing the order of the face's vertices
            mesh.faces[i] = face[::-1]
    
    # Recompute face normals
    mesh.fix_normals()

    return mesh

@profile
def Vertex_test(mesh):
    for i, vertex in enumerate(mesh.vertices):
        if 4< vertex[0] <5:
            print("potential problem with index", i, vertex )
        if 3< vertex[1] <4:
            print("potential problem with index", i, vertex )
        if vertex[2] == 0:
            print("potential problem with index", i, vertex )
    
    
@profile
def truncate_planes(mesh, planes, tolerance=1e-8):
    truncated_planes = []
    for plane in planes:
        intersections = trimesh.intersections.mesh_plane(mesh, plane['normal'], plane['origin'], tolerance=tolerance)
        if intersections is not None and len(intersections) > 0:
            truncated_planes.append(plane)
    return truncated_planes

@profile
def adjust_vertices_to_planes(vertices, faces, planes, combined_directions, layer_height):
    for plane in planes:
        plane_normal = plane['normal']
        plane_origin = plane['origin']
        for i, vertex in enumerate(vertices):
            vertex_shift = layer_height * combined_directions[i]
            new_vertex = vertex + vertex_shift
            distance_to_plane = np.dot(new_vertex - plane_origin, plane_normal)
            if distance_to_plane < 0:  # If vertex is on the wrong side of the plane, adjust it
                correction = -distance_to_plane * plane_normal
                vertices[i] += correction
    return vertices
    
# Ensure direction_ratio is between 0 and 1
@profile
def OLDOnion_layer(layer_height, face_index, mesh, direction_ratio):
    if not 0 <= direction_ratio <= 1:
        raise ValueError("direction_ratio must be between 0 and 1")
    
    # Get the face normals of the mesh
    face_normals = mesh.face_normals
    
    # Get the normal vector of the specified face
    base_normal = face_normals[face_index]
    
    # Create a direction vector for each vertex by combining the base normal and the average normal of connected faces
    combined_directions = np.zeros_like(mesh.vertices)
    vertex_faces = mesh.vertex_faces
    for i, vertex in enumerate(mesh.vertices):
        # Get the faces that share this vertex
        faces_indices = vertex_faces[i][vertex_faces[i] != -1]
        if len(faces_indices) == 0:
            continue
        
        # Calculate the average normal for these faces, excluding the specified face index
        avg_normal = face_normals[faces_indices].mean(axis=0)
        # Combine the normals
        combined_direction = (1 - direction_ratio) * avg_normal + direction_ratio * base_normal
        combined_direction /= np.linalg.norm(combined_direction)  # Normalize the vector
        combined_directions[i] = combined_direction
        #if i == 0:
            #print(combined_directions[i])
    # Create a copy of the vertices to avoid accumulating shifts
    new_vertices = mesh.vertices.copy()
    #print("the 0th vertices in onion func: ",new_vertices[0])
    # Adjust the vertices of the faces except the specified face index
    shifted = np.zeros(len(new_vertices), dtype=bool)
    for i in range(len(mesh.faces)):
        if i == face_index:
            continue
        face_vertices = mesh.faces[i]
        for j in face_vertices:
            if not shifted[j]:
                
                # Shift each vertex by the layer height in the direction of the combined vector
                new_vertices[j] -= layer_height * combined_directions[j]
                shifted[j] = True
                
    # Create and return the new mesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    #print("the 0th vertices in onion func: ",new_mesh.vertices[0])
    
    return new_mesh

@profile
def Onion_layer(layer_height, face_index, mesh, direction_ratio):

    # Calculate the centroid of the mesh
    centroid = mesh.vertices.mean(axis=0)
    
    # Calculate direction from centroid to each vertex
    directions = mesh.vertices - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Avoid division by zero for vertices at or very close to centroid
    epsilon = 1e-10
    safe_norms = np.where(norms < epsilon, epsilon, norms)
    
    # Normalize directions
    normalized_directions = directions / safe_norms
    
    # Move each vertex toward centroid by exactly layer_height
    # If vertex is closer than layer_height to centroid, move it to centroid
    movement = np.minimum(norms.flatten(), layer_height)
    new_vertices = mesh.vertices - normalized_directions * movement[:, np.newaxis]
    
    # Ensure no NaN or Inf values
    new_vertices = np.nan_to_num(new_vertices, nan=centroid[0], posinf=centroid[0], neginf=centroid[0])
    
    # Create and return the new mesh
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    
    return new_mesh


@profile
def process_mesh_layers(mesh, layer_height, face_index, direction_ratio):

    all_intersection_lines = []
    Test_mesh = ensure_faces_outward(mesh.copy())
    
    show_lines(Test_mesh, False)
    
    # PRE-CALCULATE NUMBER OF LAYERS
    # Calculate the maximum distance from centroid to any vertex
    centroid = Test_mesh.vertices.mean(axis=0)
    max_distance = np.max(np.linalg.norm(Test_mesh.vertices - centroid, axis=1))
    
    # With uniform shrinking, we need max_distance / layer_height iterations
    # Plus a small buffer for the 0.1 threshold
    num_layers = int(max_distance / layer_height) + 5  # +5 for safety margin
    
    # Safety check to prevent infinite loops
    if num_layers > 1000000:
        print(f"Warning: Calculated {num_layers} layers, capping at 1000000")
        num_layers = 1000000
    
    # Process iterations until mesh is too small
    for i in range(num_layers):
        # Show lines of the current mesh
        all_intersection_lines.extend(show_lines(Test_mesh, False))
        # Apply the UNIFORM Onion Layer transformation
        Test_mesh = Onion_layer(layer_height, face_index, Test_mesh, direction_ratio)
        Test_mesh = ensure_faces_outward(Test_mesh)
        
        # Early exit if mesh becomes too small
        if (Test_mesh.bounds[1][0] - Test_mesh.bounds[0][0] < 0.1 or
            Test_mesh.bounds[1][1] - Test_mesh.bounds[0][1] < 0.1 or
            Test_mesh.bounds[1][2] - Test_mesh.bounds[0][2] < 0.1):
            break
    
    show_lines(Test_mesh, False)
    return all_intersection_lines, i + 1


# Main processing loop
for index, Test_mesh in enumerate(meshes):
    with profile_block(f"process_mesh_{index}"):
        # Process mesh layers using optimized function
        direction_ratio = 0
        face_index = 2
        
        # Use the optimized process_mesh_layers function
        all_intersection_lines, iter_count = process_mesh_layers(
            Test_mesh, layer_height, face_index, direction_ratio
        )
        
        print(f"Processed mesh {index} with {iter_count} iterations")
        output_filename = f"all_intersection_lines_{index}.txt"
        
        with profile_block(f"write_output_file_{index}"):
            with open(os.path.join(output_folder, output_filename), 'w') as file:
                # Write the header
                file.write("lines output Version=0.1\n")
                file.write("LineX_0, LineY_0, LineZ_0, LineA_0, LineB_0, LineC_0, LineX_1, LineY_1, LineZ_1, LineA_1, LineB_1, LineC_1\n")
                
                # Write the intersection lines data
                for lines in all_intersection_lines:
                    for line in lines:
                        for i in range(len(line) - 1):
                            start_point = line[i].flatten()  # Ensures it's a 1D array
                            end_point = line[i + 1].flatten()  # Ensures it's a 1D array
                            file.write(f"{float(start_point[0]):.6f}, {float(start_point[1]):.6f}, {float(start_point[2]):.6f}, {float(start_point[3]):.6f}, {float(start_point[4]):.6f}, {float(start_point[5]):.6f}, "
                                       f"{float(end_point[0]):.6f}, {float(end_point[1]):.6f}, {float(end_point[2]):.6f}, {float(end_point[3]):.6f}, {float(end_point[4]):.4f}, {float(end_point[5]):.6f}\n")

        print(f"Saved intersection lines for mesh {index} to {output_filename}")

# Generate profiling report if debug mode is enabled
if ProfilerManager.is_debug_mode():
    profiling_folder = os.path.join(output_folder, "Profiling")
    os.makedirs(profiling_folder, exist_ok=True)
    ProfilerManager.print_report(os.path.join(profiling_folder, "Profiling_Onion3d.txt"))
    ProfilerManager.save_csv_report(os.path.join(profiling_folder, "Profiling_Onion3d.csv"))

