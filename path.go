//go:build path
// +build path

package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

func main() {
	// Initialize profiler from environment
	InitFromEnv()

	// Define the output folder path
	outputFolder := "DecompositionOUTPUT"
	adjacencyFile := filepath.Join(outputFolder, "adjacency_list.txt")

	// Load the adjacency list
	var adjacencyList [][2]int
	if _, err := os.Stat(adjacencyFile); os.IsNotExist(err) {
		fmt.Printf("Error: %s does not exist.\n", adjacencyFile)
		adjacencyList = [][2]int{}
	} else {
		data, err := os.ReadFile(adjacencyFile)
		if err != nil {
			fmt.Printf("Error reading adjacency file: %v\n", err)
			adjacencyList = [][2]int{}
		} else {
			lines := strings.Split(string(data), "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					i, err1 := strconv.Atoi(parts[0])
					j, err2 := strconv.Atoi(parts[1])
					if err1 == nil && err2 == nil {
						adjacencyList = append(adjacencyList, [2]int{i, j})
					}
				}
			}
		}
		fmt.Println("Adjacency List:")
		fmt.Println(adjacencyList)
	}

	// Create a map to easily access the connections for each node
	connectionsDict := make(map[int][]int)
	for _, pair := range adjacencyList {
		i, j := pair[0], pair[1]
		if _, exists := connectionsDict[i]; !exists {
			connectionsDict[i] = []int{}
		}
		if _, exists := connectionsDict[j]; !exists {
			connectionsDict[j] = []int{}
		}
		connectionsDict[i] = append(connectionsDict[i], j)
		connectionsDict[j] = append(connectionsDict[j], i)
	}

	// Count the connections for each node
	connectionCount := make(map[int]int)
	for _, pair := range adjacencyList {
		connectionCount[pair[0]]++
		connectionCount[pair[1]]++
	}

	// Sort the nodes based on the number of connections, descending
	type nodeCount struct {
		node  int
		count int
	}
	var sortedNodes []nodeCount
	for node, count := range connectionCount {
		sortedNodes = append(sortedNodes, nodeCount{node, count})
	}
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].count > sortedNodes[j].count
	})
	fmt.Println("Sorted Nodes by Connection Count:")
	fmt.Println(sortedNodes)

	// Load the decomposed meshes
	var meshes []*Mesh
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

	// Calculate centers
	Profile("calculate_centers", func() {
		_ = calculateCenters(meshes)
	})

	var startNode int
	if len(sortedNodes) > 0 {
		startNode = sortedNodes[0].node
	} else {
		startNode = 0
	}

	// Create ordered list
	var orderedList []int
	Profile("create_ordered_list", func() {
		orderedList = createOrderedListByConvexHull(connectionsDict, meshes, startNode)
	})

	// Save ordered list
	Profile("save_ordered_list", func() {
		file, err := os.Create(filepath.Join(outputFolder, "ordered_list.txt"))
		if err != nil {
			fmt.Printf("Error creating ordered list file: %v\n", err)
			os.Exit(1)
		}
		defer file.Close()

		for _, node := range orderedList {
			fmt.Fprintf(file, "%d\n", node)
		}
	})

	fmt.Println("Ordered List of Nodes saved:")
	fmt.Println(orderedList)

	// Generate profiling report if debug mode is enabled
	if globalProfiler.IsDebugMode() {
		profilingFolder := filepath.Join(outputFolder, "Profiling")
		err := os.MkdirAll(profilingFolder, 0755)
		if err != nil {
			fmt.Printf("Error creating profiling folder: %v\n", err)
			return
		}
		globalProfiler.PrintReport(filepath.Join(profilingFolder, "Profiling_Path.txt"))
		globalProfiler.SaveCSVReport(filepath.Join(profilingFolder, "Profiling_Path.csv"))
	}
}

// calculateCenters calculates the centroids of meshes
func calculateCenters(meshes []*Mesh) []Vec3 {
	centers := make([]Vec3, len(meshes))
	for i, mesh := range meshes {
		centers[i] = mesh.Centroid()
	}
	return centers
}

// createOrderedListByConvexHull creates an ordered list based on convex hull volume
func createOrderedListByConvexHull(connectionsDict map[int][]int, meshes []*Mesh, startNode int) []int {
	orderedList := []int{}
	visited := make(map[int]bool)

	// Start with the specified node
	currentNode := startNode
	orderedList = append(orderedList, currentNode)
	visited[currentNode] = true

	for len(orderedList) < len(meshes) {
		minHullVolume := float32(math.MaxFloat32)
		nextNode := -1

		// Get all points of the current convex hull
		var currentPoints []Vec3
		for _, node := range orderedList {
			currentPoints = append(currentPoints, meshes[node].Vertices...)
		}
		_ = computeConvexHull(currentPoints)

		// Check neighbors of nodes in ordered list
		for _, node := range orderedList {
			neighbors := connectionsDict[node]
			for _, neighbor := range neighbors {
				if !visited[neighbor] {
					neighborPoints := meshes[neighbor].Vertices
					newPoints := append(currentPoints, neighborPoints...)
					newHull := computeConvexHull(newPoints)
					hullVolume := ConvexHullVolume(newHull)

					if hullVolume < minHullVolume {
						minHullVolume = hullVolume
						nextNode = neighbor
					}
				}
			}
		}

		if nextNode != -1 {
			orderedList = append(orderedList, nextNode)
			visited[nextNode] = true
		} else {
			// If no unvisited neighbors, pick the next unvisited node
			for node := range meshes {
				if !visited[node] {
					orderedList = append(orderedList, node)
					visited[node] = true
					break
				}
			}
		}
	}

	return orderedList
}

// computeConvexHull computes the convex hull using QuickHull algorithm
func computeConvexHull(points []Vec3) *Mesh {
	if len(points) < 4 {
		return nil
	}

	// Run QuickHull algorithm
	hullPoints := quickHull(points)

	// Create a mesh from the convex hull points
	// For simplicity, we'll create a triangular mesh connecting the hull points
	// In 3D, this would normally create a proper convex hull mesh
	if len(hullPoints) < 3 {
		return nil
	}

	// Create a simple triangulation of the hull points
	// This is a basic fan triangulation from the first point
	vertices := make([]Vec3, len(hullPoints))
	copy(vertices, hullPoints)

	faces := [][3]int{}
	if len(hullPoints) >= 3 {
		// Create triangular faces using fan triangulation
		for i := 1; i < len(hullPoints)-1; i++ {
			faces = append(faces, [3]int{0, i, i + 1})
		}
	}

	return &Mesh{Vertices: vertices, Faces: faces}
}

// quickHull implements the QuickHull algorithm for 3D points
func quickHull(points []Vec3) []Vec3 {
	if len(points) <= 1 {
		return points
	}

	// Find the points with minimum and maximum x-coordinates
	minX, maxX := findMinMaxX(points)

	// Split points into left and right of the line minX-maxX
	leftPoints, rightPoints := splitPoints(points, minX, maxX)

	// Recursively find hull points
	hull := []Vec3{minX, maxX}

	// Find points above the line
	hull = append(hull, findHull(leftPoints, minX, maxX)...)

	// Find points below the line
	hull = append(hull, findHull(rightPoints, maxX, minX)...)

	// Remove duplicates
	hull = removeDuplicates(hull)

	return hull
}

// findMinMaxX finds points with minimum and maximum x-coordinates
func findMinMaxX(points []Vec3) (Vec3, Vec3) {
	if len(points) == 0 {
		return Vec3{}, Vec3{}
	}

	minX, maxX := points[0], points[0]
	for _, p := range points[1:] {
		if p[0] < minX[0] {
			minX = p
		}
		if p[0] > maxX[0] {
			maxX = p
		}
	}
	return minX, maxX
}

// splitPoints splits points into those left and right of the line p1-p2
func splitPoints(points []Vec3, p1, p2 Vec3) ([]Vec3, []Vec3) {
	left, right := []Vec3{}, []Vec3{}

	for _, p := range points {
		if p == p1 || p == p2 {
			continue
		}

		// Use cross product to determine side
		// For 3D points, we need to consider the plane
		// This is a simplified 2D projection approach
		cross := (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0])
		if cross > 0 {
			left = append(left, p)
		} else if cross < 0 {
			right = append(right, p)
		}
		// Points exactly on the line are ignored
	}

	return left, right
}

// findHull recursively finds hull points for one side of the line
func findHull(points []Vec3, p1, p2 Vec3) []Vec3 {
	if len(points) == 0 {
		return []Vec3{}
	}

	// Find the point farthest from the line p1-p2
	farthest := findFarthestPoint(points, p1, p2)
	if farthest == (Vec3{}) {
		return []Vec3{}
	}

	// Recursively find hull points on both sides of the new triangle
	leftPoints1, rightPoints1 := splitPoints(points, p1, farthest)
	leftPoints2, rightPoints2 := splitPoints(points, farthest, p2)

	hull := []Vec3{farthest}

	// Find hull points between p1 and farthest
	hull = append(hull, findHull(leftPoints1, p1, farthest)...)

	// Find hull points between farthest and p2
	hull = append(hull, findHull(rightPoints1, farthest, p2)...)

	// Also check the other splits
	hull = append(hull, findHull(leftPoints2, p1, farthest)...)
	hull = append(hull, findHull(rightPoints2, farthest, p2)...)

	return hull
}

// findFarthestPoint finds the point farthest from the line p1-p2
func findFarthestPoint(points []Vec3, p1, p2 Vec3) Vec3 {
	if len(points) == 0 {
		return Vec3{}
	}

	maxDist := float32(0)
	farthest := Vec3{}

	lineVec := p2.Sub(p1)
	lineLen := lineVec.Norm()

	if lineLen < 1e-6 {
		// Degenerate line, just return the first point
		return points[0]
	}

	lineVec = lineVec.Mul(1.0 / lineLen)

	for _, p := range points {
		// Calculate distance from point to line
		// Vector from p1 to p
		v := p.Sub(p1)

		// Project onto line direction
		proj := v[0]*lineVec[0] + v[1]*lineVec[1] + v[2]*lineVec[2]

		// Point on line closest to p
		closest := p1.Add(lineVec.Mul(proj))

		// Distance from p to closest point on line
		distVec := p.Sub(closest)
		dist := distVec.Norm()

		if dist > maxDist {
			maxDist = dist
			farthest = p
		}
	}

	return farthest
}

// removeDuplicates removes duplicate points from the hull
func removeDuplicates(points []Vec3) []Vec3 {
	seen := make(map[Vec3]bool)
	result := []Vec3{}

	for _, p := range points {
		if !seen[p] {
			seen[p] = true
			result = append(result, p)
		}
	}

	return result
}

// computeBounds computes the bounding box of points
func computeBounds(points []Vec3) (min, max Vec3) {
	if len(points) == 0 {
		return Vec3{0, 0, 0}, Vec3{0, 0, 0}
	}
	min = points[0]
	max = points[0]
	for _, p := range points {
		if p[0] < min[0] {
			min[0] = p[0]
		}
		if p[1] < min[1] {
			min[1] = p[1]
		}
		if p[2] < min[2] {
			min[2] = p[2]
		}
		if p[0] > max[0] {
			max[0] = p[0]
		}
		if p[1] > max[1] {
			max[1] = p[1]
		}
		if p[2] > max[2] {
			max[2] = p[2]
		}
	}
	return min, max
}
