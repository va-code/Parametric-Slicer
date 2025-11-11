# TODO list
# Profiling Analysis vs TODO File

## Executive Summary

**Total Onion3d execution time: 14,161 seconds (~3.9 hours)**

The profiling data **strongly confirms** the TODO file's priorities, with some important nuances:

### ✅ Confirmed: Onion3d is the bottleneck (65.6% of tracked time)
### ✅ Confirmed: The three main functions are the problem
### ⚠️  Nuance: The loop optimization is partially done, but intersection calculations dominate

---

## Profiling Results Breakdown

### Top 3 Bottlenecks (65.6% of total time):

1. **`process_mesh_layers`**: 3,199s (22.6%)
   - 66 calls (one per mesh)
   - Average: 48.5 seconds per mesh
   - Range: 2.1s to 357s per mesh

2. **`show_lines`**: 3,073s (21.7%)
   - 9,834 calls (very frequent!)
   - Average: 312ms per call
   - Range: 0.66ms to 46.2s (huge variance!)

3. **`calculate_intersection_lines`**: 3,023s (21.3%)
   - 19,668 calls (most frequent function!)
   - Average: 154ms per call
   - Range: 0.21ms to 44.1s (huge variance!)

### Other Notable Functions:

- **`ensure_faces_outward`**: 122s (0.86%) - 9,768 calls, avg 12.5ms
- **`create_planes`**: 46s - 19,668 calls, avg 2.3ms (already fast!)
- **`Onion_layer`**: 3.8s - 9,702 calls, avg 0.39ms (already optimized!)

---

## Comparison with TODO File

### ✅ TODO Priority #1: "Optimize Onion3d.py Infinite Loop" (10-100x speedup)

**Status: PARTIALLY ADDRESSED**

The TODO file correctly identified this as the #1 priority. However, the profiling shows:

1. **The loop structure has been optimized** (now using `process_mesh_layers` with pre-calculated layers)
   - The TODO suggested replacing the while loop with a for loop based on pre-calculated layers
   - **This has been implemented!** (see lines 356-371 in Onion3d.py)
   - However, the function still takes 22.6% of total time

2. **But the internal operations are still slow**:
   - `show_lines()` is called 9,834 times and takes 21.7% of total time
   - `calculate_intersection_lines()` is called 19,668 times and takes 21.3% of total time
   
3. **The TODO's specific solutions are still relevant**:
   - ✅ "Pre-calculate number of layers" - **DONE** (lines 356-363)
   - ❌ "Cache plane intersections" - **NOT DONE** (19,668 calls suggests no caching)
   - ❌ "Vectorize operations" - **PARTIALLY DONE** (Onion_layer is fast, but intersections aren't)
   - ❌ "Adaptive layer heights" - **NOT DONE**

### Key Insight from Profiling:

**The loop itself is optimized, but the operations INSIDE the loop are the real bottleneck.**

The TODO estimated 10-100x speedup, but we're only seeing partial gains because:
- The loop structure is better (for loop vs while loop)
- But `show_lines()` and `calculate_intersection_lines()` are still being called thousands of times with no caching

---

## Detailed Analysis

### 1. `show_lines()` - 21.7% of total time, 9,834 calls

**What it does:**
- Creates primary and secondary planes
- Calls `calculate_intersection_lines()` for each plane
- Called twice per iteration in `process_mesh_layers` (line 373)

**Problem:**
- Called 9,834 times across 66 meshes = ~149 calls per mesh
- Each call creates new random planes (line 118-119)
- No caching of plane intersections
- Variance is huge: 0.66ms to 46.2s (suggests input-dependent performance)

**TODO Alignment:**
- The TODO mentions caching plane intersections - this is exactly what's needed here
- The TODO suggests the loop iterates millions of times - we're seeing 149 iterations per mesh on average

**Optimization Opportunities:**
1. **Cache plane intersections** - If the same planes are used, reuse results
2. **Reduce plane count** - Maybe we don't need primary AND secondary planes every iteration?
3. **Early termination** - If no intersections found, skip expensive operations

### 2. `calculate_intersection_lines()` - 21.3% of total time, 19,668 calls

**What it does:**
- Calls `trimesh.intersections.mesh_plane()` for each plane
- Processes intersection results to create oriented lines

**Problem:**
- Called 19,668 times (exactly 2x the number of `show_lines` calls)
- Each call processes a plane-mesh intersection
- `trimesh.intersections.mesh_plane()` is expensive for large meshes
- Variance is huge: 0.21ms to 44.1s

**TODO Alignment:**
- The TODO doesn't specifically mention this function, but it's the core operation
- The TODO suggests vectorization - mesh-plane intersections are hard to vectorize, but we could:
  - Batch multiple plane queries
  - Use spatial acceleration structures
  - Reduce the number of planes checked

**Optimization Opportunities:**
1. **Reduce plane count** - Fewer planes = fewer intersection calculations
2. **Spatial acceleration** - Use BVH or octree to quickly reject planes that don't intersect
3. **Mesh simplification** - As mesh shrinks, reduce vertex count for faster intersections
4. **Parallel processing** - Process multiple planes in parallel

### 3. `process_mesh_layers()` - 22.6% of total time, 66 calls

**What it does:**
- Main loop that processes each mesh through onion layers
- Calls `show_lines()` and `Onion_layer()` repeatedly

**Problem:**
- Takes 48.5 seconds per mesh on average
- Some meshes take 357 seconds (process_mesh_34)
- Variance suggests some meshes are much more complex

**TODO Alignment:**
- ✅ Pre-calculation is done (lines 356-363)
- ✅ For loop is used instead of while loop (line 371)
- ❌ But still calls `show_lines()` every iteration (line 373)

**Optimization Opportunities:**
1. **Adaptive layer heights** - Use larger steps when far from surface
2. **Early termination** - Stop when mesh is too small
3. **Skip expensive operations** - Don't call `show_lines()` every single iteration
4. **Mesh-specific optimization** - Some meshes (like mesh_34) need special handling

---

## Individual Mesh Performance

**Worst performing meshes:**
- `process_mesh_34`: 445.8s (7.4 minutes!)
- `process_mesh_64`: 261.5s (4.4 minutes)
- `process_mesh_5`: 168.1s (2.8 minutes)
- `process_mesh_32`: 167.2s (2.8 minutes)

**Best performing meshes:**
- `process_mesh_44`: 2.6s
- `process_mesh_42`: 2.5s
- `process_mesh_58`: 6.7s

**Insight:** Some meshes are 170x slower than others! This suggests:
- Mesh complexity varies wildly
- Some meshes might have degenerate geometry
- Adaptive algorithms could help

---

## Recommendations Based on Profiling

### Immediate Actions (High Impact, Easy):

1. **Cache plane intersections** (TODO priority #1, solution #2)
   - If the same plane is checked multiple times, cache the result
   - Estimated speedup: 2-5x for `calculate_intersection_lines`

2. **Reduce `show_lines()` call frequency** (TODO priority #1, optimization)
   - Don't call `show_lines()` every single iteration
   - Maybe every Nth iteration, or use adaptive frequency
   - Estimated speedup: 2-10x reduction in calls

3. **Optimize `calculate_intersection_lines()`** (New priority)
   - Add early termination if no intersections found
   - Use spatial acceleration for plane rejection
   - Estimated speedup: 2-5x

### Medium-Term Actions (High Impact, Medium Difficulty):

4. **Adaptive layer heights** (TODO priority #1, solution #4)
   - Use larger steps when far from surface
   - Smaller steps near surface
   - Estimated speedup: 2-3x

5. **Mesh simplification during processing** (New idea)
   - As mesh shrinks, simplify it to reduce vertex count
   - Faster intersections on simpler meshes
   - Estimated speedup: 2-5x for later iterations

### Long-Term Actions (High Impact, Hard):

6. **Parallel processing** (New idea)
   - Process multiple meshes in parallel
   - Process multiple planes in parallel
   - Estimated speedup: 4-8x (on multi-core systems)

---

## Expected Speedup After Optimizations

**Current:** 14,161 seconds (~3.9 hours)

**After implementing TODO priorities:**
- Caching plane intersections: 2-5x → **7,080-2,832 seconds**
- Reducing show_lines calls: 2-10x → **3,540-283 seconds**
- Optimizing intersections: 2-5x → **1,770-57 seconds**

**Combined (multiplicative):** 10-50x speedup → **1,416-283 seconds (24 minutes - 5 minutes)**

This aligns with TODO file's estimate: "10-50x faster" → "30-60 minutes → 1-5 minutes"

---

## Conclusion

✅ **The TODO file is accurate and well-prioritized**

The profiling data confirms:
1. Onion3d is indeed the bottleneck (65.6% of time)
2. The loop optimization (TODO #1) is partially done but needs more work
3. The specific solutions suggested (caching, vectorization) are still needed
4. The estimated speedup (10-100x) is achievable if we implement the remaining optimizations

**Next Steps:**
1. Implement plane intersection caching (TODO #1, solution #2)
2. Reduce `show_lines()` call frequency
3. Optimize `calculate_intersection_lines()` with spatial acceleration
4. Add adaptive layer heights

After these optimizations, we should see the 10-50x speedup predicted in the TODO file.










---old---


This document outlines performance optimizations for the Parametric Slicer codebase, ranked by estimated impact.

## Priority Rankings

### 🔴 **CRITICAL - Highest Impact**

---

## 1. Optimize Onion3d.py Infinite Loop (10-100x speedup)

**File:** `Onion3d.py`  
**Lines:** 293-306  
**Estimated Speedup:** 10-100x  
**Difficulty:** Medium

### Problem
The while loop can iterate millions of times, each iteration:
- Calling `show_lines()` which creates planes and calculates intersections
- Shrinking the mesh with `Onion_layer()`
- Recalculating face normals with `ensure_faces_outward()`

### Current Code
```python
while (Test_mesh.bounds[1][0] - Test_mesh.bounds[0][0] >= 0.1 and
       Test_mesh.bounds[1][1] - Test_mesh.bounds[0][1] >= 0.1 and
       Test_mesh.bounds[1][2] - Test_mesh.bounds[0][2] >= 0.1):
    all_intersection_lines.extend(show_lines(Test_mesh, False))
    Test_mesh = Onion_layer(layer_height, 2, Test_mesh, direction_ratio)
    Test_mesh = ensure_faces_outward(Test_mesh)
```

### Solutions
1. **Pre-calculate number of layers**
   ```python
   max_dimension = max(
       Test_mesh.bounds[1][0] - Test_mesh.bounds[0][0],
       Test_mesh.bounds[1][1] - Test_mesh.bounds[0][1],
       Test_mesh.bounds[1][2] - Test_mesh.bounds[0][2]
   )
   num_layers = int((max_dimension - 0.1) / layer_height)
   
   for i in range(num_layers):
       # Process layer
   ```

2. **Cache plane intersections** - Don't recalculate the same planes

3. **Vectorize operations** - Process all vertices at once instead of loops

4. **Adaptive layer heights** - Use larger steps when far from surface

---

## 2. Fix Adjaceny.py KDTree Inefficiency (10-50x speedup)

**File:** `Adjaceny.py`  
**Lines:** 36-50  
**Estimated Speedup:** 10-50x  
**Difficulty:** Easy

### Problem
KDTrees are created but then every vertex is checked individually in Python loops.

### Current Code
```python
def check_mesh_adjacency(mesh1, mesh2, threshold=1e-3):
    mesh1_kdtree = KDTree(mesh1.vertices)
    mesh2_kdtree = KDTree(mesh2.vertices)
    
    for vertex in mesh1.vertices:  # ❌ Slow loop
        distance, _ = mesh2_kdtree.query(vertex)
        if distance < threshold:
            return True
    
    for vertex in mesh2.vertices:  # ❌ Another slow loop
        distance, _ = mesh1_kdtree.query(vertex)
        if distance < threshold:
            return True
    return False
```

### Optimized Code
```python
def check_mesh_adjacency(mesh1, mesh2, threshold=1e-3):
    mesh2_kdtree = KDTree(mesh2.vertices)
    
    # Query all mesh1 vertices at once
    distances, _ = mesh2_kdtree.query(mesh1.vertices, k=1)
    
    # Check if any distance is below threshold
    return np.any(distances < threshold)
```

**Why it's faster:** Vectorized NumPy operations are 10-100x faster than Python loops.

---

## 3. Optimize path.py Distance Matrix (5-20x speedup)

**File:** `path.py`  
**Lines:** 138  
**Estimated Speedup:** 5-20x  
**Difficulty:** Easy

### Problem
`distance.cdist()` creates a full NxM distance matrix in memory. For 10K vertex meshes, this is 100 million calculations!

### Current Code
```python
dist_matrix = distance.cdist(current_points, neighbor_points, 'euclidean')
closest_dist = dist_matrix.min()
```

**Memory usage:** For 10K vertices × 10K vertices = 800MB per comparison!

### Optimized Code
```python
from scipy.spatial import KDTree

neighbor_kdtree = KDTree(neighbor_points)
distances, _ = neighbor_kdtree.query(current_points, k=1)
closest_dist = distances.min()
```

**Memory usage:** ~80KB for the same meshes

---

## 4. Optimize path.py Convex Hull Computation (5-10x speedup)

**File:** `path.py`  
**Lines:** 222-236  
**Estimated Speedup:** 5-10x  
**Difficulty:** Medium

### Problem
Convex hull is recalculated for every neighbor in the inner loop.

### Current Code
```python
current_points = np.vstack([meshes[node].vertices for node in ordered_list])
current_hull = trimesh.convex.convex_hull(current_points)  # ❌ Expensive

for neighbor in connections_dict.get(current_node, []):
    if neighbor not in visited:
        neighbor_points = meshes[neighbor].vertices
        new_points = np.vstack([current_hull.vertices, neighbor_points])
        new_hull = trimesh.convex.convex_hull(new_points)  # ❌ Very expensive
        hull_volume = new_hull.volume
```

### Solutions
1. **Pre-compute convex hulls** for each mesh
   ```python
   mesh_hulls = [trimesh.convex.convex_hull(m.vertices) for m in meshes]
   ```

2. **Use bounding boxes for filtering** before expensive convex hull
   ```python
   # Quick rejection test with AABB
   if not boxes_overlap(current_box, neighbor_box):
       continue
   # Then do expensive convex hull
   ```

3. **Cache intermediate results** - don't recompute the same hulls

---

### 🟡 **MODERATE - Medium Impact**

---

## 5. Implement Mesh Caching (2-3x I/O speedup)

**Files:** Multiple  
**Estimated Speedup:** 2-3x for I/O operations  
**Difficulty:** Easy

### Problem
The same STL files are loaded 4+ times across different scripts:
- `Adjaceny.py` line 21
- `path.py` line 61
- `Onion3d.py` line 20
- `graphing.py` line 24

### Solution
Modify `main.py` to load meshes once and pass them:

```python
def load_meshes(output_folder):
    """Load all decomposed meshes once."""
    mesh_files = sorted([f for f in os.listdir(output_folder) 
                        if f.startswith("mesh_") and f.endswith(".stl")])
    return [trimesh.load(os.path.join(output_folder, f), force="mesh") 
            for f in mesh_files]

def main():
    # Load meshes once
    meshes = load_meshes("DecompositionOUTPUT")
    
    # Pass to each script instead of reloading
    run_adjacency(meshes)
    run_path_planning(meshes)
    run_onion3d(meshes)
```

---

### 🟢 **LOW PRIORITY - UX Improvement**

---

## 6. Add Progress Indicators (No performance gain, better UX)

**Files:** All Python scripts  
**Estimated Speedup:** 0x (but better user experience)  
**Difficulty:** Very Easy

### Solution
Add `tqdm` for progress bars:

```python
from tqdm import tqdm

# In decomposer.py
for idx, part in tqdm(enumerate(meshes), desc="Saving meshes"):
    # ...

# In Adjaceny.py
for i in tqdm(range(len(meshes)), desc="Checking adjacency"):
    # ...

# In Onion3d.py
for index, Test_mesh in tqdm(enumerate(meshes), desc="Processing meshes"):
    # ...
```

---

## Implementation Order

1. **Start with #2 (Adjaceny.py)** - Easiest and high impact
2. **Then #3 (path.py cdist)** - Also easy, high impact
3. **Then #1 (Onion3d.py)** - Biggest impact but more complex
4. **Then #4 (path.py convex hull)** - Medium difficulty
5. **Then #5 (mesh caching)** - Requires refactoring main.py
6. **Finally #6 (progress bars)** - Nice to have

---

## Expected Overall Speedup

For complex meshes (>10K vertices, 20+ decomposed parts):
- **Before optimizations:** 30-60 minutes
- **After optimizations:** 1-5 minutes
- **Total speedup:** 10-50x faster

---

## Testing

After each optimization:
1. Test with `A.stl` (simple case)
2. Test with `Stanford dragon.stl` (complex case)
3. Verify output files are identical
4. Time the execution with `time python3 main.py`

