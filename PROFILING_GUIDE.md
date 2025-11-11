# Performance Profiling Guide

This guide explains how to use the built-in performance profiling system to identify bottlenecks in the Parametric Slicer codebase.

## Quick Start

To enable profiling, simply run the main pipeline with the `--debug` flag:

```bash
python3 main.py --debug
```

This will:
1. Enable performance tracking for all instrumented functions
2. Collect timing data and call counts throughout the pipeline
3. Generate detailed profiling reports for each stage

## Output Files

When debug mode is enabled, profiling reports are generated in the `DecompositionOUTPUT/` folder:

- `decomposer_profiling.txt` - Profiling report for mesh decomposition stage
- `decomposer_profiling.csv` - CSV export for spreadsheet analysis
- `Adjaceny_profiling.txt` - Profiling report for adjacency computation stage
- `Adjaceny_profiling.csv` - CSV export
- `path_profiling.txt` - Profiling report for path ordering stage
- `path_profiling.csv` - CSV export
- `Onion3d_profiling.txt` - Profiling report for onion layer processing (typically the longest)
- `Onion3d_profiling.csv` - CSV export
- `profiling_report.txt` - Combined report for the entire pipeline

## Understanding the Reports

Each profiling report contains several sections:

### 1. Summary Statistics
```
Total execution time tracked: 45.231s
Total function calls tracked: 12,847
```

### 2. Top Functions by Total Time
Shows which functions are consuming the most total time:
```
Function Name                              Calls    Total(s)      Avg(ms)      Min(ms)      Max(ms)
--------------------------------------------------------------------------------
Onion3d.show_lines                           127      32.451      255.520        1.234      512.345
Onion3d.calculate_intersection_lines         254      18.234       71.784        0.456      234.567
...
```

**Key metrics:**
- **Calls**: Number of times the function was called
- **Total(s)**: Total time spent in this function (seconds)
- **Avg(ms)**: Average time per call (milliseconds)
- **Min/Max(ms)**: Fastest and slowest individual calls

### 3. Time Distribution Analysis
Shows what percentage of total runtime each function consumes:
```
Function Name                              % of Total    Cumulative %
--------------------------------------------------------------------------------
Onion3d.show_lines                              71.72%          71.72%
Onion3d.calculate_intersection_lines            40.31%          82.03%
...
```

This helps identify the "80/20 rule" - often 20% of functions consume 80% of runtime.

### 4. Most Frequently Called Functions
Shows functions that are called most often (even if they're individually fast):
```
Function Name                              Call Count     Avg Time(ms)
--------------------------------------------------------------------------------
Onion3d.Onion_layer                              1,234           2.345
calculate_intersection_lines                       567          32.123
...
```

### 5. Potential Bottlenecks
Automatically identifies functions that are either:
- High total time (long-running)
- High call count (frequently called)
- Both

These are your primary optimization targets.

## Interpreting Results

### What to Optimize

1. **High Total Time + Low Call Count** = Individual calls are slow
   - Optimize the algorithm itself
   - Look for expensive operations (nested loops, matrix operations)
   - Consider vectorization or parallel processing

2. **High Total Time + High Call Count** = Called too often AND slow
   - Reduce number of calls (caching, memoization)
   - Optimize the function itself
   - Consider refactoring to batch operations

3. **Low Total Time + Very High Call Count** = Death by a thousand cuts
   - Cache results if function is pure/deterministic
   - Inline small functions if appropriate
   - Batch multiple calls into one

### Example Analysis

From the example output:
```
• Onion3d.show_lines
  - Total time: 32.451s (71.7% of total)
  - Calls: 127
  - Avg time: 255.5ms
```

**Interpretation**: `show_lines()` is called 127 times and takes over 70% of total runtime. Each call averages 255ms, which is quite slow. This is your primary bottleneck.

**Optimization strategies:**
- Reduce the number of calls (do we need to call it 127 times?)
- Optimize the function itself (vectorize operations, reduce allocations)
- Profile deeper into this function to find internal bottlenecks

## Advanced Usage

### Custom Profiling in Your Code

You can add profiling to your own functions:

```python
from profiler import profile, profile_block

# Decorate a function
@profile
def my_function():
    # your code here
    pass

# Profile a code block
with profile_block("custom_operation"):
    # code to profile
    pass
```

### Detailed Profiling

For extra detailed logging of individual function calls:

```python
@profile(detailed=True)
def my_function(arg1, arg2):
    pass
```

This prints each call as it happens:
```
[PROFILE] my_function(arg1=value1, arg2=value2) took 12.345ms
```

### Running Without main.py

You can also run individual scripts with profiling:

```bash
DEBUG=1 python3 Onion3d.py
```

Or in Python code:
```python
from profiler import ProfilerManager
ProfilerManager.set_debug_mode(True)
```

### CSV Export for Analysis

All reports are also exported as CSV files that you can open in Excel, Google Sheets, or analyze with pandas:

```python
import pandas as pd

df = pd.read_csv('DecompositionOUTPUT/Onion3d_profiling.csv')
print(df.describe())
```

## Common Bottlenecks in 3D Processing

Based on typical 3D mesh processing workloads, watch for:

1. **Mesh-Plane Intersections** (`trimesh.intersections.mesh_plane`)
   - Can be very slow for high-poly meshes
   - Consider reducing plane count or simplifying mesh

2. **Distance Computations** (KDTree queries, vertex comparisons)
   - O(n²) or O(n log n) operations
   - Use spatial data structures (KDTree, BVH)

3. **Mesh Operations** (copy, transform, boolean operations)
   - Hidden allocations and copies
   - Reuse buffers when possible

4. **I/O Operations** (loading/saving STL files)
   - Consider binary format instead of ASCII
   - Batch operations instead of many small writes

5. **Repeated Calculations** (normals, centroids, bounds)
   - Cache results if mesh doesn't change
   - Use lazy evaluation

## Performance Tips

1. **Start with the biggest bottleneck** - Focus on functions consuming >10% of total time
2. **Measure before and after** - Run with `--debug` before and after optimization
3. **Profile on realistic data** - Use actual production meshes, not toy examples
4. **Consider algorithmic improvements first** - Going from O(n²) to O(n log n) beats micro-optimizations
5. **Check call counts** - Sometimes the best optimization is calling a function less often

## Troubleshooting

**Q: Debug mode makes my code too slow**
A: Profiling adds ~5-10% overhead. For very tight loops, this can be noticeable. Profile on representative data, then disable for production.

**Q: Not all functions appear in the report**
A: Only functions decorated with `@profile` or code in `profile_block()` contexts are tracked. Add decorators to functions you want to profile.

**Q: The CSV file won't open**
A: Make sure to use UTF-8 encoding. Most modern spreadsheet programs handle this automatically.

**Q: Function times seem inaccurate**
A: Nested profiling can cause confusion. The total time for a parent function includes time spent in child functions. Use the report's breakdown to understand the hierarchy.

## Example Workflow

1. **Initial profiling**:
   ```bash
   python3 main.py --debug
   ```

2. **Identify bottleneck** (e.g., `show_lines()` takes 70% of time)

3. **Add more detailed profiling** to that function:
   ```python
   @profile
   def show_lines(Mesh, GRAPH):
       with profile_block("create_primary_planes"):
           primary_planes = create_planes(...)
       
       with profile_block("calculate_primary_intersections"):
           primary_intersection_lines = calculate_intersection_lines(...)
       # ...
   ```

4. **Re-profile** to see which part of `show_lines()` is slow

5. **Optimize** the specific bottleneck

6. **Verify improvement**:
   ```bash
   python3 main.py --debug
   # Compare new report to old report
   ```

## Getting Help

If you're unsure how to interpret results or optimize a particular bottleneck, include:
- The relevant section of the profiling report
- The mesh characteristics (vertex count, face count)
- Hardware specs (CPU, RAM)

Happy optimizing! 🚀

