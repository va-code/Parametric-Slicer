import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_intersection_lines(mesh, planes):
    intersection_lines = []
    for plane in planes:
        intersections = trimesh.intersections.mesh_plane(mesh, plane['normal'], plane['origin'])
        if intersections is not None and len(intersections) > 0:
            oriented_lines = []
            for line in intersections:
                for i in range(len(line) - 1):
                    start_point = line[i]
                    end_point = line[i + 1]
                    
                    line_direction = end_point - start_point
                    line_direction /= np.linalg.norm(line_direction)
                    
                    normal_vector = np.cross(plane['normal'], line_direction)
                    normal_vector /= np.linalg.norm(normal_vector)
                    #debugging issue with normalization of normal vector is not working 
                    if (normal_vector[0]*normal_vector[0] +normal_vector[1]*normal_vector[1] +normal_vector[2]*normal_vector[2]) > 1.05:
                        print("Normal Vect", normal_vector)
                        print("Start Point", start_point)
                        print("End Point", end_point)
                        quit()
                    oriented_lines.append(np.concatenate([start_point, normal_vector]))
                    oriented_lines.append(np.concatenate([end_point, normal_vector]))
                    
            intersection_lines.append(np.array(oriented_lines))

    return intersection_lines
    
# Initialize the Base Vector in a random direction
base_vector = np.random.rand(3) - 0.5
base_vector /= np.linalg.norm(base_vector)  # Normalize the vector

# Function to show lines
def show_lines(Mesh, GRAPH):
    # Store all primary and secondary intersection lines
    all_intersection_lines = []
    
    # Create a random 3D vector
    random_vector = np.random.rand(3) - 0.5
    random_vector /= np.linalg.norm(random_vector)  # Normalize the vector
    print(f"Random direction vector: {random_vector}")

    # Create primary planes and calculate intersections
    primary_planes = create_planes(Mesh, random_vector, layer_height)
    print(f"Number of primary planes created: {len(primary_planes)}")
    primary_intersection_lines = calculate_intersection_lines(Mesh, primary_planes)
    print(f"Number of primary intersection lines: {len(primary_intersection_lines)}")
    all_intersection_lines.append(primary_intersection_lines)

    # Create secondary planes (90 degrees offset) and calculate intersections
    orthogonal_vector = np.cross(random_vector, [1, 0, 0])
    if np.linalg.norm(orthogonal_vector) == 0:  # Handle the case where the random vector is parallel to [1, 0, 0]
        orthogonal_vector = np.cross(random_vector, [0, 1, 0])
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)

    secondary_planes = create_planes(Mesh, orthogonal_vector, layer_height)
    print(f"Number of secondary planes created: {len(secondary_planes)}")
    secondary_intersection_lines = calculate_intersection_lines(Mesh, secondary_planes)
    print(f"Number of secondary intersection lines: {len(secondary_intersection_lines)}")
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

def Vertex_test(mesh):
    for i, vertex in enumerate(mesh.vertices):
        if 4< vertex[0] <5:
            print("potential problem with index", i, vertex )
        if 3< vertex[1] <4:
            print("potential problem with index", i, vertex )
        if vertex[2] == 0:
            print("potential problem with index", i, vertex )
    
    

def truncate_planes(mesh, planes, tolerance=1e-8):
    truncated_planes = []
    for plane in planes:
        intersections = trimesh.intersections.mesh_plane(mesh, plane['normal'], plane['origin'], tolerance=tolerance)
        if intersections is not None and len(intersections) > 0:
            truncated_planes.append(plane)
    return truncated_planes

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

def Onion_layer(layer_height, face_index, mesh, direction_ratio):
    # Ensure the layer height is a negative value for shrinking
    #layer_height = -0.2  # Set the scaling factor for shrinking by 0.2mm

    # Calculate the centroid of the mesh
    centroid = mesh.vertices.mean(axis=0)
    
    # Calculate the scaling factor relative to the centroid
    scale_factors = np.ones_like(mesh.vertices) * -layer_height
    scaled_vertices = mesh.vertices + (mesh.vertices - centroid) * scale_factors
    
    # Create and return the new mesh
    new_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=mesh.faces)
    
    return new_mesh

for index, Test_mesh in enumerate(meshes):
    # Store all intersection lines
    all_intersection_lines = []

    # Loop to apply Onion Layer transformation until the mesh's tallest point is less than 0.1
    direction_ratio = 0
    Test_mesh = ensure_faces_outward(Test_mesh)

    #print("Test_mesh \n",Test_mesh)
    #print("[1][2] \n",Test_mesh.bounds[1][2])
    #print("[0][2] \n",Test_mesh.bounds[0][2])
    show_lines(Test_mesh, False)
    iter = 0
    while (Test_mesh.bounds[1][0] - Test_mesh.bounds[0][0] >= 0.1 and
           Test_mesh.bounds[1][1] - Test_mesh.bounds[0][1] >= 0.1 and
           Test_mesh.bounds[1][2] - Test_mesh.bounds[0][2] >= 0.1):
        # Show lines of the current mesh
        all_intersection_lines.extend(show_lines(Test_mesh, False))
        # Apply the Onion Layer transformation
        #Vertex_test(Test_mesh)
        Test_mesh = Onion_layer(layer_height, 2, Test_mesh, direction_ratio)
        Test_mesh = ensure_faces_outward(Test_mesh)
        
        if iter == 1000000:
            print("LIKELY ERROR, THROWING ERROR BECAUSE LOOPED 1,000,000 TIMES TRYING TO GENERATE ALL LINES FOR THE SUBMESH INDEX: ", index)
            break
        iter += 1
        
    # Output the collected intersection lines
    #print("Collected intersection lines:", all_intersection_lines)
    show_lines(Test_mesh, False)
    output_filename = f"all_intersection_lines_{index}.txt"
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

