#!/usr/bin/env python3
"""
Profiling and debugging module for Parametric Slicer.

This module provides decorators and utilities to track function execution times,
call counts, and generate performance reports to identify bottlenecks.

Usage:
    from profiler import profile, ProfilerManager
    
    # Enable debug mode
    ProfilerManager.set_debug_mode(True)
    
    # Decorate functions to profile
    @profile
    def my_function():
        pass
    
    # At end of execution, print report
    ProfilerManager.print_report()
"""

import time
import functools
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, Any
import sys

class ProfilerManager:
    """
    Central manager for profiling data collection and reporting.
    """
    _debug_mode = False
    _call_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'count': 0,
        'total_time': 0.0,
        'min_time': float('inf'),
        'max_time': 0.0,
        'times': []
    })
    _call_stack: List[Tuple[str, float]] = []
    _enabled = True
    
    @classmethod
    def set_debug_mode(cls, enabled: bool):
        """Enable or disable debug mode."""
        cls._debug_mode = enabled
        cls._enabled = enabled
        if enabled:
            print("="*70)
            print("DEBUG MODE ENABLED - Performance profiling active")
            print("="*70)
    
    @classmethod
    def is_debug_mode(cls) -> bool:
        """Check if debug mode is enabled."""
        return cls._debug_mode
    
    @classmethod
    def record_call(cls, func_name: str, execution_time: float, args_info: str = ""):
        """Record a function call and its execution time."""
        if not cls._enabled:
            return
            
        stats = cls._call_stats[func_name]
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        stats['times'].append(execution_time)
        
        if args_info:
            if 'args_info' not in stats:
                stats['args_info'] = []
            stats['args_info'].append(args_info)
    
    @classmethod
    def push_call(cls, func_name: str):
        """Push a function call onto the stack (for nested profiling)."""
        if cls._enabled:
            cls._call_stack.append((func_name, time.time()))
    
    @classmethod
    def pop_call(cls) -> Tuple[str, float]:
        """Pop a function call from the stack and return elapsed time."""
        if cls._enabled and cls._call_stack:
            func_name, start_time = cls._call_stack.pop()
            elapsed = time.time() - start_time
            return func_name, elapsed
        return "", 0.0
    
    @classmethod
    def get_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get all collected statistics."""
        return dict(cls._call_stats)
    
    @classmethod
    def reset(cls):
        """Reset all profiling data."""
        cls._call_stats.clear()
        cls._call_stack.clear()
    
    @classmethod
    def print_report(cls, output_file: str = None):
        """
        Print a formatted profiling report.
        
        Args:
            output_file: Optional file path to save the report
        """
        if not cls._call_stats:
            print("\nNo profiling data collected.")
            return
        
        # Sort by total time (descending)
        sorted_stats = sorted(
            cls._call_stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        # Build report
        lines = []
        lines.append("\n" + "="*90)
        lines.append("PERFORMANCE PROFILING REPORT")
        lines.append("="*90)
        lines.append("")
        
        # Summary statistics
        total_time = sum(stat['total_time'] for _, stat in sorted_stats)
        total_calls = sum(stat['count'] for _, stat in sorted_stats)
        lines.append(f"Total execution time tracked: {total_time:.3f}s")
        lines.append(f"Total function calls tracked: {total_calls}")
        lines.append("")
        
        # Detailed function statistics
        lines.append("TOP FUNCTIONS BY TOTAL TIME:")
        lines.append("-"*90)
        lines.append(f"{'Function Name':<40} {'Calls':>8} {'Total(s)':>12} {'Avg(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}")
        lines.append("-"*90)
        
        for func_name, stats in sorted_stats[:30]:  # Top 30 functions
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            lines.append(
                f"{func_name:<40} "
                f"{stats['count']:>8} "
                f"{stats['total_time']:>12.3f} "
                f"{avg_time*1000:>12.3f} "
                f"{stats['min_time']*1000:>12.3f} "
                f"{stats['max_time']*1000:>12.3f}"
            )
        
        lines.append("-"*90)
        lines.append("")
        
        # Time distribution analysis
        lines.append("TIME DISTRIBUTION ANALYSIS:")
        lines.append("-"*90)
        lines.append(f"{'Function Name':<40} {'% of Total':>15} {'Cumulative %':>15}")
        lines.append("-"*90)
        
        cumulative_percent = 0.0
        for func_name, stats in sorted_stats[:20]:  # Top 20 for distribution
            percent = (stats['total_time'] / total_time * 100) if total_time > 0 else 0
            cumulative_percent += percent
            lines.append(
                f"{func_name:<40} "
                f"{percent:>14.2f}% "
                f"{cumulative_percent:>14.2f}%"
            )
        
        lines.append("-"*90)
        lines.append("")
        
        # Call frequency analysis
        lines.append("MOST FREQUENTLY CALLED FUNCTIONS:")
        lines.append("-"*90)
        sorted_by_calls = sorted(
            cls._call_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        lines.append(f"{'Function Name':<40} {'Call Count':>15} {'Avg Time(ms)':>15}")
        lines.append("-"*90)
        
        for func_name, stats in sorted_by_calls[:20]:  # Top 20 by call count
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            lines.append(
                f"{func_name:<40} "
                f"{stats['count']:>15} "
                f"{avg_time*1000:>15.3f}"
            )
        
        lines.append("-"*90)
        lines.append("")
        
        # Bottleneck identification
        lines.append("POTENTIAL BOTTLENECKS:")
        lines.append("-"*90)
        lines.append("(Functions with high total time OR high call count)")
        lines.append("")
        
        bottlenecks = []
        for func_name, stats in cls._call_stats.items():
            score = stats['total_time'] + (stats['count'] / 1000.0)  # Weighted score
            bottlenecks.append((func_name, stats, score))
        
        bottlenecks.sort(key=lambda x: x[2], reverse=True)
        
        for func_name, stats, score in bottlenecks[:10]:
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            lines.append(f"  • {func_name}")
            lines.append(f"    - Total time: {stats['total_time']:.3f}s ({stats['total_time']/total_time*100:.1f}% of total)")
            lines.append(f"    - Calls: {stats['count']}")
            lines.append(f"    - Avg time: {avg_time*1000:.3f}ms")
            lines.append("")
        
        lines.append("="*90)
        
        # Print to console
        report = "\n".join(lines)
        print(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")
    
    @classmethod
    def save_csv_report(cls, output_file: str = "profiling_report.csv"):
        """Save profiling data as CSV for further analysis."""
        import csv
        
        if not cls._call_stats:
            print("No profiling data to save.")
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Function Name', 'Call Count', 'Total Time (s)', 
                'Average Time (ms)', 'Min Time (ms)', 'Max Time (ms)'
            ])
            
            for func_name, stats in sorted(
                cls._call_stats.items(), 
                key=lambda x: x[1]['total_time'], 
                reverse=True
            ):
                avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                writer.writerow([
                    func_name,
                    stats['count'],
                    f"{stats['total_time']:.6f}",
                    f"{avg_time*1000:.6f}",
                    f"{stats['min_time']*1000:.6f}",
                    f"{stats['max_time']*1000:.6f}"
                ])
        
        print(f"CSV report saved to: {output_file}")


def profile(func: Callable = None, *, detailed: bool = False) -> Callable:
    """
    Decorator to profile function execution time and call count.
    
    Args:
        func: The function to profile
        detailed: If True, also log each individual call
    
    Usage:
        @profile
        def my_function():
            pass
        
        @profile(detailed=True)
        def my_detailed_function():
            pass
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not ProfilerManager.is_debug_mode():
                # If debug mode is off, just call the function normally
                return f(*args, **kwargs)
            
            # Record function name with module
            func_name = f"{f.__module__}.{f.__name__}" if hasattr(f, '__module__') else f.__name__
            
            # Build args info if detailed mode
            args_info = ""
            if detailed:
                args_repr = []
                if args:
                    args_repr.extend([repr(a)[:50] for a in args[:3]])  # First 3 args
                if kwargs:
                    args_repr.extend([f"{k}={repr(v)[:30]}" for k, v in list(kwargs.items())[:3]])
                args_info = ", ".join(args_repr)
            
            # Time the function
            start_time = time.time()
            ProfilerManager.push_call(func_name)
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                ProfilerManager.pop_call()
                execution_time = time.time() - start_time
                ProfilerManager.record_call(func_name, execution_time, args_info)
                
                if detailed:
                    print(f"[PROFILE] {func_name}({args_info}) took {execution_time*1000:.3f}ms")
        
        return wrapper
    
    # Support both @profile and @profile(detailed=True)
    if func is None:
        return decorator
    else:
        return decorator(func)


def profile_block(name: str):
    """
    Context manager for profiling code blocks.
    
    Usage:
        with profile_block("my_code_block"):
            # code to profile
            pass
    """
    class ProfileBlock:
        def __init__(self, block_name):
            self.block_name = block_name
            self.start_time = None
        
        def __enter__(self):
            if ProfilerManager.is_debug_mode():
                self.start_time = time.time()
                ProfilerManager.push_call(self.block_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if ProfilerManager.is_debug_mode() and self.start_time:
                ProfilerManager.pop_call()
                execution_time = time.time() - self.start_time
                ProfilerManager.record_call(self.block_name, execution_time)
            return False
    
    return ProfileBlock(name)


# Convenience function to enable debug mode from environment variable
def init_from_env():
    """Initialize profiler based on DEBUG environment variable."""
    debug_enabled = os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes')
    ProfilerManager.set_debug_mode(debug_enabled)


# Auto-initialize from environment if DEBUG is set
if os.environ.get('DEBUG'):
    init_from_env()

