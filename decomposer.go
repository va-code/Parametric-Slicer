//go:build decomposer
// +build decomposer

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	// Initialize profiler from environment
	InitFromEnv()

	// Get input file from environment variable or use default
	inputFile := os.Getenv("INPUT_FILE")
	if inputFile == "" {
		inputFile = "A.stl"
	}
	fmt.Printf("Processing input file: %s\n", inputFile)

	outputFolder := "DecompositionOUTPUT"

	// Call Python decomposer.py to do the actual convex decomposition
	// The Python script uses coacd with decimation=False to ensure no mesh simplification
	// It will handle the coacd decomposition and save mesh_*.stl files
	Profile("run_python_decomposer", func() {
		fmt.Println("Calling Python decomposer.py for convex decomposition...")

		// Set up environment variables for the Python script
		env := os.Environ()
		// Remove existing DEBUG and INPUT_FILE if present
		newEnv := []string{}
		for _, e := range env {
			if !strings.HasPrefix(e, "DEBUG=") && !strings.HasPrefix(e, "INPUT_FILE=") {
				newEnv = append(newEnv, e)
			}
		}
		env = newEnv

		// Pass the debug and input file settings to Python
		if os.Getenv("DEBUG") == "1" {
			env = append(env, "DEBUG=1")
		}
		env = append(env, "INPUT_FILE="+inputFile)

		// Run the Python decomposer script with virtual environment activated
		cmd := exec.Command("bash", "-c", "source .venv/bin/activate && python3 decomposer.py")
		cmd.Env = env
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		err := cmd.Run()
		if err != nil {
			fmt.Printf("Error running Python decomposer: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("Python decomposition completed successfully")
	})

	// Load the decomposed meshes created by Python
	var meshes []*Mesh
	Profile("load_decomposed_meshes", func() {
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

		for _, meshFile := range meshFiles {
			mesh, err := LoadSTL(filepath.Join(outputFolder, meshFile))
			if err != nil {
				fmt.Printf("Error loading decomposed mesh %s: %v\n", meshFile, err)
				continue
			}
			meshes = append(meshes, mesh)
		}

		if len(meshes) == 0 {
			fmt.Println("Warning: No decomposed meshes found. Python decomposition may have failed.")
			// Try to load the original mesh as fallback
			originalMesh, err := LoadSTL(inputFile)
			if err != nil {
				fmt.Printf("Error loading original mesh as fallback: %v\n", err)
				os.Exit(1)
			}
			meshes = []*Mesh{originalMesh}
		}
	})

	// Verify decomposed meshes (already exported by Python)
	Profile("verify_decomposed_meshes", func() {
		fmt.Printf("Verification: Found %d decomposed mesh parts\n", len(meshes))

		// Ensure output folder exists (should already exist from Python)
		err := os.MkdirAll(outputFolder, 0755)
		if err != nil {
			fmt.Printf("Error creating output folder: %v\n", err)
			os.Exit(1)
		}

		// List the mesh files that were created by Python
		files, err := os.ReadDir(outputFolder)
		if err != nil {
			fmt.Printf("Error reading output folder: %v\n", err)
			return
		}

		var meshFiles []string
		for _, file := range files {
			name := file.Name()
			if strings.HasPrefix(name, "mesh_") && strings.HasSuffix(name, ".stl") {
				meshFiles = append(meshFiles, name)
			}
		}

		fmt.Printf("Decomposed meshes saved by Python: %v\n", meshFiles)
	})

	// Generate profiling report if debug mode is enabled
	if globalProfiler.IsDebugMode() {
		profilingFolder := filepath.Join(outputFolder, "Profiling")
		err := os.MkdirAll(profilingFolder, 0755)
		if err != nil {
			fmt.Printf("Error creating profiling folder: %v\n", err)
			return
		}
		globalProfiler.PrintReport(filepath.Join(profilingFolder, "Profiling_Decomposer.txt"))
		globalProfiler.SaveCSVReport(filepath.Join(profilingFolder, "Profiling_Decomposer.csv"))
	}
}
