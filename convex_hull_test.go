//go:build test
// +build test

package main

import (
	"fmt"
	"testing"
)

// TestQuickHull3D tests the QuickHull implementation
func TestQuickHull3D(t *testing.T) {
	// Simple test points forming a tetrahedron
	points := []Vec3{
		{0, 0, 0},
		{1, 0, 0},
		{0.5, 1, 0},
		{0.5, 0.5, 1},
		{0.5, 0.5, 0.5}, // Interior point
	}

	hull := quickHull3D(points)

	// Should have 4 points (tetrahedron vertices)
	if len(hull) != 4 {
		t.Errorf("Expected 4 hull points, got %d", len(hull))
	}

	fmt.Printf("Convex hull points: %v\n", hull)
}

// TestConvexHullFunction tests the ConvexHull function
func TestConvexHullFunction(t *testing.T) {
	points := []Vec3{
		{0, 0, 0},
		{1, 0, 0},
		{0.5, 1, 0},
		{0.5, 0.5, 1},
	}

	mesh := ConvexHull(points)

	if mesh == nil {
		t.Error("ConvexHull returned nil")
		return
	}

	if len(mesh.Vertices) < 3 {
		t.Errorf("Expected at least 3 vertices, got %d", len(mesh.Vertices))
	}

	fmt.Printf("Convex hull mesh: %d vertices, %d faces\n", len(mesh.Vertices), len(mesh.Faces))
}
