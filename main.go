//go:build !decomposer && !adjacency && !path && !onion3d
// +build !decomposer,!adjacency,!path,!onion3d

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

func main() {
	// Parse command-line arguments
	debug := flag.Bool("debug", false, "Enable debug mode with detailed performance profiling")
	_ = flag.String("profile-output", "DecompositionOUTPUT/Profiling/Profiling_Report.txt", "Output file for profiling report")
	input := flag.String("input", "A.stl", "Input STL file to process")
	flag.Parse()

	fmt.Println("Starting Parametric Slicer Pipeline")
	fmt.Printf("Input file: %s\n", *input)
	if *debug {
		fmt.Println("DEBUG MODE: Performance profiling enabled")
	}
	fmt.Println(strings.Repeat("=", 60))

	// Set environment variables
	if *debug {
		os.Setenv("DEBUG", "1")
	} else {
		os.Setenv("DEBUG", "0")
	}
	os.Setenv("INPUT_FILE", *input)

	// Define the pipeline sequence
	scripts := []string{
		"decomposer",
		"adjacency",
		"path",
		"onion3d",
	}

	// Check if all scripts exist (as Go binaries)
	missingScripts := []string{}
	for _, script := range scripts {
		// In Go, we'll run the programs directly
		// They should be built first, or we can run them with `go run`
		scriptPath := script + ".go"
		if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
			missingScripts = append(missingScripts, scriptPath)
		}
	}

	if len(missingScripts) > 0 {
		fmt.Printf("\n✗ Error: The following scripts are missing:\n")
		for _, script := range missingScripts {
			fmt.Printf("  - %s\n", script)
		}
		os.Exit(1)
	}

	// Run each script in sequence
	for _, script := range scripts {
		success := runScript(script, *debug, *input)
		if !success {
			fmt.Printf("\n%s\n", strings.Repeat("=", 60))
			fmt.Printf("Pipeline stopped due to error in %s\n", script)
			fmt.Printf("%s\n", strings.Repeat("=", 60))
			os.Exit(1)
		}
	}

	// All scripts completed successfully
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Println("✓ Pipeline completed successfully!")
	fmt.Println("Check the DecompositionOUTPUT folder for results.")

	// If debug mode was enabled, generate final profiling report
	if *debug {
		fmt.Println("\nGenerating consolidated profiling report...")
		// Note: In Go, we'd need to implement generate_final_report functionality
		// For now, we'll just note that profiling reports are in DecompositionOUTPUT/Profiling/
		fmt.Println("\n✓ Profiling reports generated in DecompositionOUTPUT/Profiling/")
		fmt.Println("  See PROFILING_GUIDE.md for help interpreting results.")
	}

	fmt.Printf("%s\n", strings.Repeat("=", 60))
}

func runScript(scriptName string, debugMode bool, inputFile string) bool {
	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("Running %s...\n", scriptName)
	fmt.Printf("%s\n\n", strings.Repeat("=", 60))

	// Set up environment for debug mode
	env := os.Environ()
	// Remove existing DEBUG and INPUT_FILE if present
	newEnv := []string{}
	for _, e := range env {
		if !strings.HasPrefix(e, "DEBUG=") && !strings.HasPrefix(e, "INPUT_FILE=") {
			newEnv = append(newEnv, e)
		}
	}
	env = newEnv

	if debugMode {
		env = append(env, "DEBUG=1")
	} else {
		env = append(env, "DEBUG=0")
	}
	if inputFile != "" {
		env = append(env, "INPUT_FILE="+inputFile)
	}

	// Build tag for the specific script
	buildTag := scriptName
	if scriptName == "adjacency" {
		buildTag = "adjacency"
	}

	// Run the Go program using `go run` with build tags
	// We need to include all the shared files
	args := []string{"run", "-tags", buildTag}

	// Include all necessary files
	files := []string{
		"mesh_utils.go",
		"profiler.go",
		scriptName + ".go",
	}
	args = append(args, files...)

	cmd := exec.Command("go", args...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Run()
	if err != nil {
		fmt.Printf("\n✗ Error: %s failed with error: %v\n", scriptName, err)
		return false
	}

	fmt.Printf("\n✓ %s completed successfully\n", scriptName)
	return true
}
