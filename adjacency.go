//go:build adjacency
// +build adjacency

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	// Initialize profiler from environment
	InitFromEnv()

	// Define the output folder path
	outputFolder := "DecompositionOUTPUT"
	outputAdjacencyFile := filepath.Join(outputFolder, "adjacency_list.txt")

	// Ensure the output folder exists
	if _, err := os.Stat(outputFolder); os.IsNotExist(err) {
		fmt.Printf("Error: %s does not exist.\n", outputFolder)
		os.Exit(1)
	}

	// Load the decomposed meshes
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

		// Sort files to ensure correct order
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

	// Check adjacency for each pair of meshes
	var adjacencyList [][2]int
	Profile("compute_adjacency", func() {
		for i := 0; i < len(meshes); i++ {
			for j := i + 1; j < len(meshes); j++ {
				isAdjacent := checkMeshAdjacency(meshes[i], meshes[j], 1e-3)
				if isAdjacent {
					adjacencyList = append(adjacencyList, [2]int{i, j})
				}
			}
		}
	})

	adjacencyList = removeDuplicateConnections(adjacencyList)
	fmt.Println("Meshes are adjacent:", adjacencyList)

	// Save adjacency list to file
	Profile("save_adjacency_list", func() {
		file, err := os.Create(outputAdjacencyFile)
		if err != nil {
			fmt.Printf("Error creating adjacency file: %v\n", err)
			os.Exit(1)
		}
		defer file.Close()

		for _, pair := range adjacencyList {
			fmt.Fprintf(file, "%d %d\n", pair[0], pair[1])
		}
	})

	fmt.Printf("Adjacency list saved to %s\n", outputAdjacencyFile)

	// Generate profiling report if debug mode is enabled
	if globalProfiler.IsDebugMode() {
		profilingFolder := filepath.Join(outputFolder, "Profiling")
		err := os.MkdirAll(profilingFolder, 0755)
		if err != nil {
			fmt.Printf("Error creating profiling folder: %v\n", err)
			return
		}
		globalProfiler.PrintReport(filepath.Join(profilingFolder, "Profiling_Adjaceny.txt"))
		globalProfiler.SaveCSVReport(filepath.Join(profilingFolder, "Profiling_Adjaceny.csv"))
	}
}

// removeDuplicateConnections removes duplicate connections
func removeDuplicateConnections(connections [][2]int) [][2]int {
	uniqueMap := make(map[string]bool)
	var filteredConnections [][2]int

	for _, conn := range connections {
		// Sort the pair to create a unique key
		var key string
		if conn[0] < conn[1] {
			key = fmt.Sprintf("%d,%d", conn[0], conn[1])
		} else {
			key = fmt.Sprintf("%d,%d", conn[1], conn[0])
		}

		if !uniqueMap[key] {
			uniqueMap[key] = true
			if conn[0] < conn[1] {
				filteredConnections = append(filteredConnections, [2]int{conn[0], conn[1]})
			} else {
				filteredConnections = append(filteredConnections, [2]int{conn[1], conn[0]})
			}
		}
	}

	return filteredConnections
}

// checkMeshAdjacency checks if two meshes are adjacent using KDTree
func checkMeshAdjacency(mesh1, mesh2 *Mesh, threshold float32) bool {
	var result bool
	Profile("check_mesh_adjacency", func() {
		// Build KDTree for mesh2
		tree2 := NewKDTree(mesh2.Vertices)

		// Check if any vertex of mesh1 is close to mesh2
		for _, vertex := range mesh1.Vertices {
			_, dist := tree2.Query(vertex)
			if dist < threshold {
				result = true
				return
			}
		}

		// Build KDTree for mesh1
		tree1 := NewKDTree(mesh1.Vertices)

		// Check if any vertex of mesh2 is close to mesh1
		for _, vertex := range mesh2.Vertices {
			_, dist := tree1.Query(vertex)
			if dist < threshold {
				result = true
				return
			}
		}
	})
	return result
}
