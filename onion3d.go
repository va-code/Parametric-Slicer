//go:build onion3d
// +build onion3d

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func main() {
	// Initialize profiler from environment
	InitFromEnv()

	// Define the output folder path
	outputFolder := "DecompositionOUTPUT"

	// Ensure the output folder exists
	if _, err := os.Stat(outputFolder); os.IsNotExist(err) {
		fmt.Printf("Error: %s does not exist.\n", outputFolder)
		os.Exit(1)
	}

	// Load the decomposed meshes
	layerHeight := float32(0.1)
	var meshes []*Mesh

	Profile("load_meshes", func() {
		files, err := os.ReadDir(outputFolder)
		if err != nil {
			fmt.Printf("Error reading output folder: %v\n", err)
			os.Exit(1)
		}

		var meshFiles []string
		for _, file := range files {
			name := file.Name()
			if strings.HasPrefix(name, "mesh_") && strings.HasSuffix(name, ".stl") {
				meshFiles = append(meshFiles, name)
			}
		}
		sort.Strings(meshFiles)

		for _, meshFile := range meshFiles {
			mesh, err := LoadSTL(filepath.Join(outputFolder, meshFile))
			if err != nil {
				fmt.Printf("Error loading mesh %s: %v\n", meshFile, err)
				continue
			}
			meshes = append(meshes, mesh)
		}
	})

	// Main processing loop
	Profile("process_all_meshes", func() {
		for index, testMesh := range meshes {
			// Process mesh layers using optimized function
			directionRatio := float32(0)
			faceIndex := 2

			// Use the optimized processMeshLayers function
			allIntersectionLines, iterCount := processMeshLayers(
				testMesh, layerHeight, faceIndex, directionRatio,
			)

			fmt.Printf("Processed mesh %d with %d iterations\n", index, iterCount)
			outputFilename := fmt.Sprintf("all_intersection_lines_%d.txt", index)

			Profile(fmt.Sprintf("write_output_file_%d", index), func() {
				file, err := os.Create(filepath.Join(outputFolder, outputFilename))
				if err != nil {
					fmt.Printf("Error creating output file: %v\n", err)
					return
				}
				defer file.Close()

				// Write the header
				file.WriteString("lines output Version=0.1\n")
				file.WriteString("LineX_0, LineY_0, LineZ_0, LineA_0, LineB_0, LineC_0, LineX_1, LineY_1, LineZ_1, LineA_1, LineB_1, LineC_1\n")

				// Write the intersection lines data
				for _, lines := range allIntersectionLines {
					for i := 0; i < len(lines)-1; i++ {
						startPoint := lines[i]
						endPoint := lines[i+1]
						fmt.Fprintf(file, "%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n",
							startPoint[0], startPoint[1], startPoint[2], startPoint[3], startPoint[4], startPoint[5],
							endPoint[0], endPoint[1], endPoint[2], endPoint[3], endPoint[4], endPoint[5])
					}
				}
			})

			fmt.Printf("Saved intersection lines for mesh %d to %s\n", index, outputFilename)
		}
	})

	// Generate profiling report if debug mode is enabled
	if globalProfiler.IsDebugMode() {
		profilingFolder := filepath.Join(outputFolder, "Profiling")
		err := os.MkdirAll(profilingFolder, 0755)
		if err != nil {
			fmt.Printf("Error creating profiling folder: %v\n", err)
			return
		}
		globalProfiler.PrintReport(filepath.Join(profilingFolder, "Profiling_Onion3d.txt"))
		globalProfiler.SaveCSVReport(filepath.Join(profilingFolder, "Profiling_Onion3d.csv"))
	}
}

// createPlanes creates planes along a direction vector
func createPlanes(mesh *Mesh, directionVector Vec3, layerHeight float32) []Plane {
	centroid := mesh.Centroid()
	planes := []Plane{}

	// Normalize the direction vector
	directionVector = directionVector.Normalize()

	// Create planes in the positive direction
	planeOffset := float32(0)
	min, max := mesh.Bounds()
	for {
		planeOrigin := centroid.Add(directionVector.Mul(planeOffset))
		if min[2] <= planeOrigin[2] && planeOrigin[2] <= max[2] {
			planes = append(planes, Plane{Origin: planeOrigin, Normal: directionVector})
			planeOffset += layerHeight
		} else {
			break
		}
	}

	// Create planes in the negative direction
	planeOffset = -layerHeight
	for {
		planeOrigin := centroid.Add(directionVector.Mul(planeOffset))
		if min[2] <= planeOrigin[2] && planeOrigin[2] <= max[2] {
			planes = append(planes, Plane{Origin: planeOrigin, Normal: directionVector})
			planeOffset -= layerHeight
		} else {
			break
		}
	}

	return planes
}

// OrientedPoint represents a point with normal (6 values: x,y,z,a,b,c)
type OrientedPoint [6]float32

// calculateIntersectionLines calculates intersection lines between mesh and planes
func calculateIntersectionLines(mesh *Mesh, planes []Plane) [][]OrientedPoint {
	intersectionLines := [][]OrientedPoint{}
	epsilon := float32(1e-6)

	Profile("calculate_intersections", func() {
		for _, plane := range planes {
			intersections := MeshPlaneIntersection(mesh, plane)
			if len(intersections) > 0 {
				var orientedLines []OrientedPoint
				for _, line := range intersections {
					for i := 0; i < len(line)-1; i++ {
						startPoint := line[i]
						endPoint := line[i+1]

						// Calculate line direction
						lineDirection := endPoint.Sub(startPoint)
						lineDirNorm := lineDirection.Norm()

						// Skip degenerate lines
						if lineDirNorm < epsilon {
							continue
						}

						lineDirection = lineDirection.Normalize()

						// Calculate normal vector (perpendicular to both plane normal and line direction)
						normalVector := plane.Normal.Cross(lineDirection)
						normalVecNorm := normalVector.Norm()

						// Skip if cross product is zero
						if normalVecNorm < epsilon {
							// Use a default perpendicular vector instead
							if abs(lineDirection[0]) < 0.9 {
								normalVector = lineDirection.Cross(Vec3{1, 0, 0})
							} else {
								normalVector = lineDirection.Cross(Vec3{0, 1, 0})
							}
							normalVecNorm = normalVector.Norm()
						}

						normalVector = normalVector.Normalize()

						// Create oriented line points (x,y,z,a,b,c format)
						orientedLines = append(orientedLines, OrientedPoint{
							startPoint[0], startPoint[1], startPoint[2],
							normalVector[0], normalVector[1], normalVector[2],
						})
						orientedLines = append(orientedLines, OrientedPoint{
							endPoint[0], endPoint[1], endPoint[2],
							normalVector[0], normalVector[1], normalVector[2],
						})
					}
				}

				if len(orientedLines) > 0 {
					intersectionLines = append(intersectionLines, orientedLines)
				}
			}
		}
	})

	return intersectionLines
}

// showLines shows intersection lines for a mesh
func showLines(mesh *Mesh, graph bool) [][]OrientedPoint {
	allIntersectionLines := [][]OrientedPoint{}

	Profile("create_random_vector", func() {
		// Create a random 3D vector
		rand.Seed(time.Now().UnixNano())
		randomVector := Vec3{
			float32(rand.Float64()) - 0.5,
			float32(rand.Float64()) - 0.5,
			float32(rand.Float64()) - 0.5,
		}
		randomVector = randomVector.Normalize()

		// Create primary planes and calculate intersections
		primaryPlanes := createPlanes(mesh, randomVector, 0.1)
		primaryIntersectionLines := calculateIntersectionLines(mesh, primaryPlanes)
		allIntersectionLines = append(allIntersectionLines, primaryIntersectionLines...)

		// Create secondary planes (90 degrees offset) and calculate intersections
		orthogonalVector := randomVector.Cross(Vec3{1, 0, 0})
		if orthogonalVector.Norm() == 0 {
			orthogonalVector = randomVector.Cross(Vec3{0, 1, 0})
		}
		orthogonalVector = orthogonalVector.Normalize()

		secondaryPlanes := createPlanes(mesh, orthogonalVector, 0.1)
		secondaryIntersectionLines := calculateIntersectionLines(mesh, secondaryPlanes)
		allIntersectionLines = append(allIntersectionLines, secondaryIntersectionLines...)
	})

	return allIntersectionLines
}

// onionLayer applies an onion layer transformation to a mesh
func onionLayer(layerHeight float32, faceIndex int, mesh *Mesh, directionRatio float32) *Mesh {
	// Calculate the centroid of the mesh
	centroid := mesh.Centroid()

	// Calculate direction from centroid to each vertex
	newVertices := make([]Vec3, len(mesh.Vertices))
	epsilon := float32(1e-10)

	for i, vertex := range mesh.Vertices {
		direction := vertex.Sub(centroid)
		norm := direction.Norm()

		// Avoid division by zero
		safeNorm := norm
		if safeNorm < epsilon {
			safeNorm = epsilon
		}

		// Normalize direction
		normalizedDirection := direction.Mul(1.0 / safeNorm)

		// Move each vertex toward centroid by exactly layerHeight
		// If vertex is closer than layerHeight to centroid, move it to centroid
		movement := layerHeight
		if norm < layerHeight {
			movement = norm
		}
		newVertex := vertex.Sub(normalizedDirection.Mul(movement))

		// Ensure no NaN or Inf values
		if math.IsNaN(float64(newVertex[0])) || math.IsInf(float64(newVertex[0]), 0) {
			newVertex[0] = centroid[0]
		}
		if math.IsNaN(float64(newVertex[1])) || math.IsInf(float64(newVertex[1]), 0) {
			newVertex[1] = centroid[1]
		}
		if math.IsNaN(float64(newVertex[2])) || math.IsInf(float64(newVertex[2]), 0) {
			newVertex[2] = centroid[2]
		}

		newVertices[i] = newVertex
	}

	// Create and return the new mesh
	return &Mesh{Vertices: newVertices, Faces: mesh.Faces}
}

// processMeshLayers processes mesh layers with onion transformation
func processMeshLayers(mesh *Mesh, layerHeight float32, faceIndex int, directionRatio float32) ([][]OrientedPoint, int) {
	allIntersectionLines := [][]OrientedPoint{}
	testMesh := mesh.EnsureFacesOutward()

	Profile("show_initial_lines", func() {
		showLines(testMesh, false)
	})

	// PRE-CALCULATE NUMBER OF LAYERS
	var numLayers int
	Profile("calculate_layers", func() {
		// Calculate the maximum distance from centroid to any vertex
		centroid := testMesh.Centroid()
		maxDistance := float32(0)
		for _, vertex := range testMesh.Vertices {
			dist := vertex.Sub(centroid).Norm()
			if dist > maxDistance {
				maxDistance = dist
			}
		}

		// With uniform shrinking, we need maxDistance / layerHeight iterations
		// Plus a small buffer for the 0.1 threshold
		numLayers = int(maxDistance/layerHeight) + 5

		// Safety check to prevent infinite loops
		if numLayers > 1000000 {
			fmt.Printf("Warning: Calculated %d layers, capping at 1000000\n", numLayers)
			numLayers = 1000000
		}
	})

	// Process iterations until mesh is too small
	iterCount := 0
	Profile("process_iterations", func() {
		for i := 0; i < numLayers; i++ {
			// Show lines of the current mesh
			lines := showLines(testMesh, false)
			allIntersectionLines = append(allIntersectionLines, lines...)

			// Apply the UNIFORM Onion Layer transformation
			testMesh = onionLayer(layerHeight, faceIndex, testMesh, directionRatio)
			testMesh = testMesh.EnsureFacesOutward()

			// Early exit if mesh becomes too small
			min, max := testMesh.Bounds()
			if (max[0]-min[0] < 0.1) || (max[1]-min[1] < 0.1) || (max[2]-min[2] < 0.1) {
				break
			}
			iterCount++
		}
	})

	Profile("show_final_lines", func() {
		showLines(testMesh, false)
	})
	return allIntersectionLines, iterCount + 1
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func linesCentroidShift(centroid Vec3, pointsList []OrientedPoint, amount float64) ([][]OrientedPoint, error) {
	if pointsList == nil {
		fmt.Printf("error in linesCentroidShift pointsList is nil")
		return nil, fmt.Errorf("pointsList is nil")
	}
	if len(pointsList) < 1 {
		fmt.Printf("error in linesCentroidShift pointsList length is less than 1")
		return nil, fmt.Errorf("pointsList length is less than 1")
	}

	var shiftedList [][]OrientedPoint
	for i := 0; i < len(pointsList); i++ {
		point := pointsList[i]

		// Calculate direction from centroid to point
		dir := Vec3{
			point[0] - centroid[0],
			point[1] - centroid[1],
			point[2] - centroid[2],
		}

		// Normalize direction
		dirNorm := dir.Norm()
		if dirNorm < 1e-10 {
			// Point is at centroid, shift along default direction
			dir = Vec3{1, 0, 0}
		} else {
			dir = dir.Mul(1.0 / dirNorm)
		}

		// Shift point along the direction
		shiftedPos := Vec3{point[0], point[1], point[2]}.Add(dir.Mul(float32(amount)))

		// Create shifted oriented point (keep the same normal)
		shiftedPoint := OrientedPoint{
			shiftedPos[0], shiftedPos[1], shiftedPos[2],
			point[3], point[4], point[5],
		}

		shiftedList = append(shiftedList, []OrientedPoint{shiftedPoint})
	}

	return shiftedList, nil
}
