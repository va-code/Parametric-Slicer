# Parametric Slicer - Go Implementation

This directory contains Go implementations of the Python STL processing pipeline. Each Go program follows the exact same steps as its Python counterpart.

## Structure

- `main.go` - Main pipeline orchestrator (equivalent to `main.py`)
- `decomposer.go` - Mesh decomposition (equivalent to `decomposer.py`)
- `adjacency.go` - Mesh adjacency computation (equivalent to `Adjaceny.py`)
- `path.go` - Path ordering based on convex hull (equivalent to `path.py`)
- `onion3d.go` - Onion layer processing (equivalent to `Onion3d.py`)
- `profiler.go` - Performance profiling utilities (equivalent to `profiler.py`)
- `mesh_utils.go` - Mesh manipulation utilities

## Dependencies

The Go implementation requires:
- Go 1.21 or later
- `github.com/hschendel/stl` - STL file I/O library

## Installation

```bash
# Install dependencies
go mod download

# Or install the STL library directly
go get github.com/hschendel/stl
```

## Usage

### Running Individual Programs

Each program can be run independently using `go run` with build tags:

```bash
# Decompose STL file
go run -tags decomposer decomposer.go mesh_utils.go profiler.go

# Check mesh adjacency
go run -tags adjacency adjacency.go mesh_utils.go profiler.go

# Compute path ordering
go run -tags path path.go mesh_utils.go profiler.go

# Process onion layers
go run -tags onion3d onion3d.go mesh_utils.go profiler.go
```

Alternatively, you can build binaries first:

```bash
# Build individual programs
go build -tags decomposer -o decomposer decomposer.go mesh_utils.go profiler.go
go build -tags adjacency -o adjacency adjacency.go mesh_utils.go profiler.go
go build -tags path -o path path.go mesh_utils.go profiler.go
go build -tags onion3d -o onion3d onion3d.go mesh_utils.go profiler.go

# Then run them
./decomposer
./adjacency
./path
./onion3d
```

### Running the Full Pipeline

Run the main orchestrator:

```bash
# Default (processes A.stl)
go run main.go

# With debug/profiling enabled
go run main.go --debug

# Process a different STL file
go run main.go --input fractal.stl

# Combine options
go run main.go --input fractal.stl --debug
```

### Building Binaries

You can also build standalone binaries:

```bash
# Build all programs
go build -o decomposer decomposer.go mesh_utils.go profiler.go
go build -o adjacency adjacency.go mesh_utils.go profiler.go
go build -o path path.go mesh_utils.go profiler.go
go build -o onion3d onion3d.go mesh_utils.go profiler.go
go build -o main main.go

# Then run
./main --input A.stl --debug
```

## Environment Variables

- `DEBUG` - Set to "1", "true", or "yes" to enable profiling
- `INPUT_FILE` - Input STL file path (default: "A.stl")

## Output

All programs write to the `DecompositionOUTPUT` folder, maintaining compatibility with the Python version:
- `mesh_*.stl` - Decomposed mesh files
- `adjacency_list.txt` - Adjacency relationships
- `ordered_list.txt` - Ordered mesh list
- `all_intersection_lines_*.txt` - Intersection line data
- `Profiling/` - Performance profiling reports (when debug mode is enabled)

## Differences from Python Version

1. **Convex Decomposition**: The Go version calls the Python decomposer.py script to perform the actual coacd convex decomposition, ensuring identical behavior to the Python version.

2. **Convex Hull**: Implements QuickHull algorithm for accurate 3D convex hull computation.

2. **Convex Hull**: Implements QuickHull algorithm for accurate 3D convex hull computation.

3. **Mesh-Plane Intersection**: Implemented but may need optimization for complex meshes.

4. **Profiling**: Fully implemented and compatible with the Python version's profiling output format.

## Notes

- The Go code follows the exact same algorithmic steps as the Python code
- All file formats and output structures match the Python version
- The code is structured to be easily maintainable and extensible

