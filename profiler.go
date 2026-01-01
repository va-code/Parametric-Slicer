package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// ProfilerManager manages profiling data collection and reporting
type ProfilerManager struct {
	debugMode bool
	enabled   bool
	stats     map[string]*CallStats
	mu        sync.RWMutex
}

// CallStats holds statistics for a function call
type CallStats struct {
	Count     int
	TotalTime float64
	MinTime   float64
	MaxTime   float64
	Times     []float64
}

var globalProfiler = &ProfilerManager{
	stats: make(map[string]*CallStats),
}

// SetDebugMode enables or disables debug mode
func (pm *ProfilerManager) SetDebugMode(enabled bool) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.debugMode = enabled
	pm.enabled = enabled
	if enabled {
		fmt.Println("======================================================================")
		fmt.Println("DEBUG MODE ENABLED - Performance profiling active")
		fmt.Println("======================================================================")
	}
}

// IsDebugMode checks if debug mode is enabled
func (pm *ProfilerManager) IsDebugMode() bool {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.debugMode
}

// RecordCall records a function call and its execution time
func (pm *ProfilerManager) RecordCall(funcName string, executionTime float64) {
	if !pm.enabled {
		return
	}
	pm.mu.Lock()
	defer pm.mu.Unlock()

	stats, exists := pm.stats[funcName]
	if !exists {
		stats = &CallStats{
			MinTime: executionTime,
			MaxTime: executionTime,
		}
		pm.stats[funcName] = stats
	}

	stats.Count++
	stats.TotalTime += executionTime
	if executionTime < stats.MinTime {
		stats.MinTime = executionTime
	}
	if executionTime > stats.MaxTime {
		stats.MaxTime = executionTime
	}
	stats.Times = append(stats.Times, executionTime)
}

// GetStats returns all collected statistics
func (pm *ProfilerManager) GetStats() map[string]*CallStats {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	result := make(map[string]*CallStats)
	for k, v := range pm.stats {
		result[k] = v
	}
	return result
}

// Reset clears all profiling data
func (pm *ProfilerManager) Reset() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.stats = make(map[string]*CallStats)
}

// PrintReport prints a formatted profiling report
func (pm *ProfilerManager) PrintReport(outputFile string) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if len(pm.stats) == 0 {
		fmt.Println("\nNo profiling data collected.")
		return
	}

	// Sort by total time (descending)
	type statEntry struct {
		name  string
		stats *CallStats
	}
	var sortedStats []statEntry
	for name, stats := range pm.stats {
		sortedStats = append(sortedStats, statEntry{name, stats})
	}
	sort.Slice(sortedStats, func(i, j int) bool {
		return sortedStats[i].stats.TotalTime > sortedStats[j].stats.TotalTime
	})

	// Build report
	var lines []string
	lines = append(lines, "\n"+strings.Repeat("=", 90))
	lines = append(lines, "PERFORMANCE PROFILING REPORT")
	lines = append(lines, strings.Repeat("=", 90))
	lines = append(lines, "")

	// Summary statistics
	var totalTime float64
	var totalCalls int
	for _, entry := range sortedStats {
		totalTime += entry.stats.TotalTime
		totalCalls += entry.stats.Count
	}
	lines = append(lines, fmt.Sprintf("Total execution time tracked: %.3fs", totalTime))
	lines = append(lines, fmt.Sprintf("Total function calls tracked: %d", totalCalls))
	lines = append(lines, "")

	// Detailed function statistics
	lines = append(lines, "TOP FUNCTIONS BY TOTAL TIME:")
	lines = append(lines, strings.Repeat("-", 90))
	lines = append(lines, fmt.Sprintf("%-40s %8s %12s %12s %12s %12s", "Function Name", "Calls", "Total(s)", "Avg(ms)", "Min(ms)", "Max(ms)"))
	lines = append(lines, strings.Repeat("-", 90))

	maxEntries := 30
	if len(sortedStats) < maxEntries {
		maxEntries = len(sortedStats)
	}
	for i := 0; i < maxEntries; i++ {
		entry := sortedStats[i]
		avgTime := entry.stats.TotalTime / float64(entry.stats.Count)
		lines = append(lines, fmt.Sprintf("%-40s %8d %12.3f %12.3f %12.3f %12.3f",
			entry.name, entry.stats.Count, entry.stats.TotalTime, avgTime*1000, entry.stats.MinTime*1000, entry.stats.MaxTime*1000))
	}

	lines = append(lines, strings.Repeat("-", 90))
	lines = append(lines, "")

	// Time distribution analysis
	lines = append(lines, "TIME DISTRIBUTION ANALYSIS:")
	lines = append(lines, strings.Repeat("-", 90))
	lines = append(lines, fmt.Sprintf("%-40s %15s %15s", "Function Name", "% of Total", "Cumulative %"))
	lines = append(lines, strings.Repeat("-", 90))

	cumulativePercent := 0.0
	maxDistEntries := 20
	if len(sortedStats) < maxDistEntries {
		maxDistEntries = len(sortedStats)
	}
	for i := 0; i < maxDistEntries; i++ {
		entry := sortedStats[i]
		percent := (entry.stats.TotalTime / totalTime * 100)
		cumulativePercent += percent
		lines = append(lines, fmt.Sprintf("%-40s %14.2f%% %14.2f%%", entry.name, percent, cumulativePercent))
	}

	lines = append(lines, strings.Repeat("-", 90))
	lines = append(lines, "")

	// Call frequency analysis
	lines = append(lines, "MOST FREQUENTLY CALLED FUNCTIONS:")
	lines = append(lines, strings.Repeat("-", 90))
	sort.Slice(sortedStats, func(i, j int) bool {
		return sortedStats[i].stats.Count > sortedStats[j].stats.Count
	})
	lines = append(lines, fmt.Sprintf("%-40s %15s %15s", "Function Name", "Call Count", "Avg Time(ms)"))
	lines = append(lines, strings.Repeat("-", 90))

	maxFreqEntries := 20
	if len(sortedStats) < maxFreqEntries {
		maxFreqEntries = len(sortedStats)
	}
	for i := 0; i < maxFreqEntries; i++ {
		entry := sortedStats[i]
		avgTime := entry.stats.TotalTime / float64(entry.stats.Count)
		lines = append(lines, fmt.Sprintf("%-40s %15d %15.3f", entry.name, entry.stats.Count, avgTime*1000))
	}

	lines = append(lines, strings.Repeat("-", 90))
	lines = append(lines, "")

	// Bottleneck identification
	lines = append(lines, "POTENTIAL BOTTLENECKS:")
	lines = append(lines, strings.Repeat("-", 90))
	lines = append(lines, "(Functions with high total time OR high call count)")
	lines = append(lines, "")

	type bottleneckEntry struct {
		name  string
		stats *CallStats
		score float64
	}
	var bottlenecks []bottleneckEntry
	for _, entry := range sortedStats {
		score := entry.stats.TotalTime + (float64(entry.stats.Count) / 1000.0)
		bottlenecks = append(bottlenecks, bottleneckEntry{entry.name, entry.stats, score})
	}
	sort.Slice(bottlenecks, func(i, j int) bool {
		return bottlenecks[i].score > bottlenecks[j].score
	})

	maxBottlenecks := 10
	if len(bottlenecks) < maxBottlenecks {
		maxBottlenecks = len(bottlenecks)
	}
	for i := 0; i < maxBottlenecks; i++ {
		b := bottlenecks[i]
		avgTime := b.stats.TotalTime / float64(b.stats.Count)
		lines = append(lines, fmt.Sprintf("  • %s", b.name))
		lines = append(lines, fmt.Sprintf("    - Total time: %.3fs (%.1f%% of total)", b.stats.TotalTime, b.stats.TotalTime/totalTime*100))
		lines = append(lines, fmt.Sprintf("    - Calls: %d", b.stats.Count))
		lines = append(lines, fmt.Sprintf("    - Avg time: %.3fms", avgTime*1000))
		lines = append(lines, "")
	}

	lines = append(lines, strings.Repeat("=", 90))

	// Print to console
	report := strings.Join(lines, "\n")
	fmt.Println(report)

	// Save to file if specified
	if outputFile != "" {
		err := os.WriteFile(outputFile, []byte(report), 0644)
		if err != nil {
			fmt.Printf("Error writing report to file: %v\n", err)
		} else {
			fmt.Printf("\nReport saved to: %s\n", outputFile)
		}
	}
}

// SaveCSVReport saves profiling data as CSV
func (pm *ProfilerManager) SaveCSVReport(outputFile string) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if len(pm.stats) == 0 {
		fmt.Println("No profiling data to save.")
		return
	}

	file, err := os.Create(outputFile)
	if err != nil {
		fmt.Printf("Error creating CSV file: %v\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write([]string{
		"Function Name", "Call Count", "Total Time (s)",
		"Average Time (ms)", "Min Time (ms)", "Max Time (ms)",
	})

	// Sort by total time
	type statEntry struct {
		name  string
		stats *CallStats
	}
	var sortedStats []statEntry
	for name, stats := range pm.stats {
		sortedStats = append(sortedStats, statEntry{name, stats})
	}
	sort.Slice(sortedStats, func(i, j int) bool {
		return sortedStats[i].stats.TotalTime > sortedStats[j].stats.TotalTime
	})

	// Write data
	for _, entry := range sortedStats {
		avgTime := entry.stats.TotalTime / float64(entry.stats.Count)
		writer.Write([]string{
			entry.name,
			fmt.Sprintf("%d", entry.stats.Count),
			fmt.Sprintf("%.6f", entry.stats.TotalTime),
			fmt.Sprintf("%.6f", avgTime*1000),
			fmt.Sprintf("%.6f", entry.stats.MinTime*1000),
			fmt.Sprintf("%.6f", entry.stats.MaxTime*1000),
		})
	}

	fmt.Printf("CSV report saved to: %s\n", outputFile)
}

// Profile decorates a function to profile its execution time
func Profile(funcName string, fn func()) {
	if !globalProfiler.IsDebugMode() {
		fn()
		return
	}
	start := time.Now()
	fn()
	elapsed := time.Since(start).Seconds()
	globalProfiler.RecordCall(funcName, elapsed)
}

// ProfileBlock is a context manager for profiling code blocks
type ProfileBlock struct {
	name      string
	startTime time.Time
}

// NewProfileBlock creates a new profile block
func NewProfileBlock(name string) *ProfileBlock {
	return &ProfileBlock{name: name}
}

// Start starts profiling
func (pb *ProfileBlock) Start() {
	if globalProfiler.IsDebugMode() {
		pb.startTime = time.Now()
	}
}

// Stop stops profiling and records the time
func (pb *ProfileBlock) Stop() {
	if globalProfiler.IsDebugMode() && !pb.startTime.IsZero() {
		elapsed := time.Since(pb.startTime).Seconds()
		globalProfiler.RecordCall(pb.name, elapsed)
	}
}

// InitFromEnv initializes profiler based on DEBUG environment variable
func InitFromEnv() {
	debugEnv := os.Getenv("DEBUG")
	debugEnabled := debugEnv == "1" || debugEnv == "true" || debugEnv == "yes"
	globalProfiler.SetDebugMode(debugEnabled)
}

func init() {
	InitFromEnv()
}
