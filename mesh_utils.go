package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/hschendel/stl"
)

// Vec3 represents a 3D vector
type Vec3 [3]float32

// Mesh represents a 3D mesh with vertices and faces
type Mesh struct {
	Vertices []Vec3
	Faces    [][3]int
}

// Add adds two vectors
func (v Vec3) Add(other Vec3) Vec3 {
	return Vec3{v[0] + other[0], v[1] + other[1], v[2] + other[2]}
}

// Sub subtracts two vectors
func (v Vec3) Sub(other Vec3) Vec3 {
	return Vec3{v[0] - other[0], v[1] - other[1], v[2] - other[2]}
}

// Mul multiplies vector by scalar
func (v Vec3) Mul(scalar float32) Vec3 {
	return Vec3{v[0] * scalar, v[1] * scalar, v[2] * scalar}
}

// Dot computes dot product
func (v Vec3) Dot(other Vec3) float32 {
	return v[0]*other[0] + v[1]*other[1] + v[2]*other[2]
}

// Cross computes cross product
func (v Vec3) Cross(other Vec3) Vec3 {
	return Vec3{
		v[1]*other[2] - v[2]*other[1],
		v[2]*other[0] - v[0]*other[2],
		v[0]*other[1] - v[1]*other[0],
	}
}

// Norm computes the norm (magnitude) of the vector
func (v Vec3) Norm() float32 {
	return float32(math.Sqrt(float64(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])))
}

// Normalize normalizes the vector
func (v Vec3) Normalize() Vec3 {
	norm := v.Norm()
	if norm < 1e-10 {
		return Vec3{0, 0, 0}
	}
	return v.Mul(1.0 / norm)
}

// LoadSTL loads an STL file and converts it to a Mesh
func LoadSTL(filename string) (*Mesh, error) {
	solid, err := stl.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Convert STL triangles to vertices and faces
	vertexMap := make(map[string]int)
	var vertices []Vec3
	var faces [][3]int

	for _, triangle := range solid.Triangles {
		var face [3]int
		for i, stlVertex := range triangle.Vertices {
			key := fmt.Sprintf("%.6f,%.6f,%.6f", stlVertex[0], stlVertex[1], stlVertex[2])
			idx, exists := vertexMap[key]
			if !exists {
				idx = len(vertices)
				vertices = append(vertices, Vec3{stlVertex[0], stlVertex[1], stlVertex[2]})
				vertexMap[key] = idx
			}
			face[i] = idx
		}
		faces = append(faces, face)
	}

	return &Mesh{Vertices: vertices, Faces: faces}, nil
}

// SaveSTL saves a Mesh to an STL file
func SaveSTL(mesh *Mesh, filename string) error {
	var triangles []stl.Triangle
	for _, face := range mesh.Faces {
		var triangle stl.Triangle
		for i := 0; i < 3; i++ {
			v := mesh.Vertices[face[i]]
			triangle.Vertices[i] = stl.Vec3{v[0], v[1], v[2]}
		}
		// Calculate normal using our Vec3 operations
		v0 := mesh.Vertices[face[0]]
		v1 := mesh.Vertices[face[1]]
		v2 := mesh.Vertices[face[2]]
		normalVec := v1.Sub(v0).Cross(v2.Sub(v0)).Normalize()
		triangle.Normal = stl.Vec3{normalVec[0], normalVec[1], normalVec[2]}
		triangles = append(triangles, triangle)
	}

	solid := &stl.Solid{
		Name:      "Mesh",
		Triangles: triangles,
	}

	return solid.WriteFile(filename)
}

// Centroid computes the centroid of the mesh
func (m *Mesh) Centroid() Vec3 {
	if len(m.Vertices) == 0 {
		return Vec3{0, 0, 0}
	}
	var sum Vec3
	for _, v := range m.Vertices {
		sum = sum.Add(v)
	}
	return sum.Mul(1.0 / float32(len(m.Vertices)))
}

// Bounds computes the bounding box of the mesh
func (m *Mesh) Bounds() (min, max Vec3) {
	if len(m.Vertices) == 0 {
		return Vec3{0, 0, 0}, Vec3{0, 0, 0}
	}
	min = m.Vertices[0]
	max = m.Vertices[0]
	for _, v := range m.Vertices {
		if v[0] < min[0] {
			min[0] = v[0]
		}
		if v[1] < min[1] {
			min[1] = v[1]
		}
		if v[2] < min[2] {
			min[2] = v[2]
		}
		if v[0] > max[0] {
			max[0] = v[0]
		}
		if v[1] > max[1] {
			max[1] = v[1]
		}
		if v[2] > max[2] {
			max[2] = v[2]
		}
	}
	return min, max
}

// Copy creates a deep copy of the mesh
func (m *Mesh) Copy() *Mesh {
	vertices := make([]Vec3, len(m.Vertices))
	copy(vertices, m.Vertices)
	faces := make([][3]int, len(m.Faces))
	copy(faces, m.Faces)
	return &Mesh{Vertices: vertices, Faces: faces}
}

// FaceNormals computes face normals for all faces
func (m *Mesh) FaceNormals() []Vec3 {
	normals := make([]Vec3, len(m.Faces))
	for i, face := range m.Faces {
		v1 := m.Vertices[face[1]].Sub(m.Vertices[face[0]])
		v2 := m.Vertices[face[2]].Sub(m.Vertices[face[0]])
		normal := v1.Cross(v2).Normalize()
		normals[i] = normal
	}
	return normals
}

// EnsureFacesOutward ensures all faces point outward
func (m *Mesh) EnsureFacesOutward() *Mesh {
	centroid := m.Centroid()
	faceNormals := m.FaceNormals()

	newMesh := m.Copy()
	for i, face := range m.Faces {
		faceNormal := faceNormals[i]
		centroidToFace := m.Vertices[face[0]].Sub(centroid)
		dotProduct := faceNormal.Dot(centroidToFace)

		if dotProduct < 0 {
			// Flip the face
			newMesh.Faces[i] = [3]int{face[2], face[1], face[0]}
		}
	}

	return newMesh
}

// VertexFaces returns a map of vertex index to face indices
func (m *Mesh) VertexFaces() map[int][]int {
	vertexFaces := make(map[int][]int)
	for i, face := range m.Faces {
		for _, vIdx := range face {
			vertexFaces[vIdx] = append(vertexFaces[vIdx], i)
		}
	}
	return vertexFaces
}

// Plane represents a plane with origin and normal
type Plane struct {
	Origin Vec3
	Normal Vec3
}

// DistanceToPlane computes the distance from a point to a plane
func DistanceToPlane(point, planeOrigin, planeNormal Vec3) float32 {
	return point.Sub(planeOrigin).Dot(planeNormal)
}

// MeshPlaneIntersection computes the intersection of a mesh with a plane
func MeshPlaneIntersection(mesh *Mesh, plane Plane) [][]Vec3 {
	var intersections [][]Vec3

	// For each face, check if it intersects the plane
	for _, face := range mesh.Faces {
		v0 := mesh.Vertices[face[0]]
		v1 := mesh.Vertices[face[1]]
		v2 := mesh.Vertices[face[2]]

		d0 := DistanceToPlane(v0, plane.Origin, plane.Normal)
		d1 := DistanceToPlane(v1, plane.Origin, plane.Normal)
		d2 := DistanceToPlane(v2, plane.Origin, plane.Normal)

		// Check if face crosses the plane
		if (d0 >= 0 && d1 < 0 && d2 < 0) || (d0 < 0 && d1 >= 0 && d2 >= 0) {
			// Edge v0-v1 crosses plane
			t := -d0 / (d1 - d0)
			p1 := v0.Add(v1.Sub(v0).Mul(t))
			// Edge v0-v2 crosses plane
			t2 := -d0 / (d2 - d0)
			p2 := v0.Add(v2.Sub(v0).Mul(t2))
			intersections = append(intersections, []Vec3{p1, p2})
		} else if (d1 >= 0 && d0 < 0 && d2 < 0) || (d1 < 0 && d0 >= 0 && d2 >= 0) {
			// Edge v1-v0 crosses plane
			t := -d1 / (d0 - d1)
			p1 := v1.Add(v0.Sub(v1).Mul(t))
			// Edge v1-v2 crosses plane
			t2 := -d1 / (d2 - d1)
			p2 := v1.Add(v2.Sub(v1).Mul(t2))
			intersections = append(intersections, []Vec3{p1, p2})
		} else if (d2 >= 0 && d0 < 0 && d1 < 0) || (d2 < 0 && d0 >= 0 && d1 >= 0) {
			// Edge v2-v0 crosses plane
			t := -d2 / (d0 - d2)
			p1 := v2.Add(v0.Sub(v2).Mul(t))
			// Edge v2-v1 crosses plane
			t2 := -d2 / (d1 - d2)
			p2 := v2.Add(v1.Sub(v2).Mul(t2))
			intersections = append(intersections, []Vec3{p1, p2})
		}
	}

	return intersections
}

// ConvexHull computes the convex hull of a set of points using QuickHull algorithm
func ConvexHull(points []Vec3) *Mesh {
	if len(points) < 4 {
		return nil
	}

	// Run QuickHull algorithm
	hullPoints := quickHull3D(points)

	// Create a mesh from the convex hull points
	// For simplicity, we'll create a triangular mesh connecting the hull points
	if len(hullPoints) < 3 {
		return nil
	}

	// Create triangular faces using fan triangulation from the first point
	vertices := make([]Vec3, len(hullPoints))
	copy(vertices, hullPoints)

	faces := [][3]int{}
	if len(hullPoints) >= 3 {
		for i := 1; i < len(hullPoints)-1; i++ {
			faces = append(faces, [3]int{0, i, i + 1})
		}
	}

	return &Mesh{Vertices: vertices, Faces: faces}
}

// quickHull3D implements the QuickHull algorithm for 3D points
func quickHull3D(points []Vec3) []Vec3 {
	if len(points) <= 1 {
		return points
	}

	// Find the points with minimum and maximum x-coordinates
	minX, maxX := findMinMaxX3D(points)

	// Split points into left and right of the line minX-maxX
	leftPoints, rightPoints := splitPoints3D(points, minX, maxX)

	// Recursively find hull points
	hull := []Vec3{minX, maxX}

	// Find points above the line
	hull = append(hull, findHull3D(leftPoints, minX, maxX)...)

	// Find points below the line
	hull = append(hull, findHull3D(rightPoints, maxX, minX)...)

	// Remove duplicates
	hull = removeDuplicates3D(hull)

	return hull
}

// findMinMaxX3D finds points with minimum and maximum x-coordinates
func findMinMaxX3D(points []Vec3) (Vec3, Vec3) {
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

// splitPoints3D splits points into those left and right of the line p1-p2
func splitPoints3D(points []Vec3, p1, p2 Vec3) ([]Vec3, []Vec3) {
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

// findHull3D recursively finds hull points for one side of the line
func findHull3D(points []Vec3, p1, p2 Vec3) []Vec3 {
	if len(points) == 0 {
		return []Vec3{}
	}

	// Find the point farthest from the line p1-p2
	farthest := findFarthestPoint3D(points, p1, p2)
	if farthest == (Vec3{}) {
		return []Vec3{}
	}

	// Recursively find hull points on both sides of the new triangle
	leftPoints1, rightPoints1 := splitPoints3D(points, p1, farthest)
	leftPoints2, rightPoints2 := splitPoints3D(points, farthest, p2)

	hull := []Vec3{farthest}

	// Find hull points between p1 and farthest
	hull = append(hull, findHull3D(leftPoints1, p1, farthest)...)

	// Find hull points between farthest and p2
	hull = append(hull, findHull3D(rightPoints1, farthest, p2)...)

	// Also check the other splits
	hull = append(hull, findHull3D(leftPoints2, p1, farthest)...)
	hull = append(hull, findHull3D(rightPoints2, farthest, p2)...)

	return hull
}

// findFarthestPoint3D finds the point farthest from the line p1-p2
func findFarthestPoint3D(points []Vec3, p1, p2 Vec3) Vec3 {
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

// removeDuplicates3D removes duplicate points from the hull
func removeDuplicates3D(points []Vec3) []Vec3 {
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

// ConvexHullVolume computes the volume of a convex hull
func ConvexHullVolume(hull *Mesh) float32 {
	if hull == nil || len(hull.Faces) == 0 {
		return 0
	}

	centroid := hull.Centroid()
	var volume float32

	for _, face := range hull.Faces {
		v0 := hull.Vertices[face[0]]
		v1 := hull.Vertices[face[1]]
		v2 := hull.Vertices[face[2]]

		// Volume of tetrahedron
		vol := v0.Sub(centroid).Dot(v1.Sub(centroid).Cross(v2.Sub(centroid))) / 6.0
		volume += vol
	}

	if volume < 0 {
		volume = -volume
	}
	return volume
}

// KDTree is a simple KD-tree implementation for spatial queries
type KDTree struct {
	points []Vec3
	left   *KDTree
	right  *KDTree
	axis   int
}

// NewKDTree creates a new KD-tree from points
func NewKDTree(points []Vec3) *KDTree {
	if len(points) == 0 {
		return nil
	}

	// Copy points
	pts := make([]Vec3, len(points))
	copy(pts, points)

	return buildKDTree(pts, 0)
}

func buildKDTree(points []Vec3, depth int) *KDTree {
	if len(points) == 0 {
		return nil
	}

	axis := depth % 3

	// Sort by axis
	sort.Slice(points, func(i, j int) bool {
		return points[i][axis] < points[j][axis]
	})

	median := len(points) / 2

	node := &KDTree{
		points: []Vec3{points[median]},
		axis:   axis,
	}

	if median > 0 {
		node.left = buildKDTree(points[:median], depth+1)
	}
	if median+1 < len(points) {
		node.right = buildKDTree(points[median+1:], depth+1)
	}

	return node
}

// Query finds the nearest point to the query point
func (kdt *KDTree) Query(query Vec3) (Vec3, float32) {
	if kdt == nil || len(kdt.points) == 0 {
		return Vec3{0, 0, 0}, math.MaxFloat32
	}

	best := kdt.points[0]
	bestDist := query.Sub(best).Norm()

	kdt.queryRecursive(query, &best, &bestDist)

	return best, bestDist
}

func (kdt *KDTree) queryRecursive(query Vec3, best *Vec3, bestDist *float32) {
	if kdt == nil {
		return
	}

	point := kdt.points[0]
	dist := query.Sub(point).Norm()
	if dist < *bestDist {
		*best = point
		*bestDist = dist
	}

	axis := kdt.axis
	diff := query[axis] - point[axis]

	var near, far *KDTree
	if diff < 0 {
		near = kdt.left
		far = kdt.right
	} else {
		near = kdt.right
		far = kdt.left
	}

	if near != nil {
		near.queryRecursive(query, best, bestDist)
	}

	if far != nil && diff*diff < *bestDist**bestDist {
		far.queryRecursive(query, best, bestDist)
	}
}
