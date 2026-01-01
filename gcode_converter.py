#!/usr/bin/env python3
"""
G-code Converter for Parametric Slicer

Converts intersection line files to G-code by:
1. Reading the ordered list to determine processing order
2. Joining intersection files in the specified order
3. Converting lines to G-code movements
4. Consolidating connected segments
5. Adding feed rates that increase by 1 for each line
"""

import os
import sys
import math
from pathlib import Path
from typing import List, Tuple


def parse_ordered_list(ordered_file: str) -> List[int]:
    """Parse the ordered_list.txt to get the processing order."""
    order = []
    with open(ordered_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                order.append(int(line))
    return order


def parse_intersection_file(filepath: str) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Parse an intersection file and return list of (start_point, end_point) tuples."""
    lines = []
    with open(filepath, 'r') as f:
        # Skip header lines
        next(f)  # Skip "lines output Version=0.1"
        next(f)  # Skip column headers

        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        # Extract start and end points (X,Y,Z coordinates)
                        start = (float(parts[0]), float(parts[1]), float(parts[2]))
                        end = (float(parts[6]), float(parts[7]), float(parts[8]))
                        lines.append((start, end))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line.strip()}")
                        continue
    return lines


def consolidate_lines(lines: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Consolidate connected line segments where end of one matches start of next."""
    if not lines:
        return lines

    consolidated = [lines[0]]

    for current_start, current_end in lines[1:]:
        last_start, last_end = consolidated[-1]

        # Check if current line connects to the previous one
        if abs(last_end[0] - current_start[0]) < 1e-6 and \
           abs(last_end[1] - current_start[1]) < 1e-6 and \
           abs(last_end[2] - current_start[2]) < 1e-6:
            # Extend the last segment
            consolidated[-1] = (last_start, current_end)
        else:
            # Start a new segment
            consolidated.append((current_start, current_end))

    return consolidated


def generate_gcode(lines: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], output_file: str):
    """Generate G-code from consolidated lines with constant feed rate and distance-based extrusion."""
    with open(output_file, 'w') as f:
        # G-code header
        f.write("; Parametric Slicer G-code Output\n")
        f.write("G21 ; Set units to millimeters\n")
        f.write("G90 ; Absolute positioning\n")
        f.write("G92 E0 ; Reset extruder position\n")
        f.write("G1 Z5 F1000 ; Move to safe Z height\n")
        f.write("\n")

        feed_rate = 1000  # Constant feed rate
        filamentFeedFactor = 1.0  # Scaling factor for filament extrusion
        extrusion_amount = 0.0  # Starting extrusion amount

        for i, (start, end) in enumerate(lines):
            # Move to start point if this is the first segment or after a discontinuity
            if i == 0:
                f.write(f"G0 X{start[0]:.3f} Y{start[1]:.3f} Z{start[2]:.3f} F{feed_rate} ; Move to start point\n")

            # Calculate distance moved for this segment
            distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)

            # Update extrusion amount based on actual distance moved
            extrusion_amount += distance * filamentFeedFactor

            # Linear move to end point with constant feed rate and distance-based extrusion
            f.write(f"G1 X{end[0]:.3f} Y{end[1]:.3f} Z{end[2]:.3f} E{extrusion_amount:.3f} F{feed_rate} ; Linear move with extrusion\n")

        # G-code footer
        f.write("\n")
        f.write("G1 Z10 F1000 ; Move to safe Z height\n")
        f.write("M104 S0 ; Turn off extruder\n")
        f.write("M30 ; End of program\n")


def main():
    """Main function to convert intersection files to G-code."""
    output_dir = "DecompositionOUTPUT"

    # Check if output directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found!")
        sys.exit(1)

    ordered_file = os.path.join(output_dir, "ordered_list.txt")
    if not os.path.exists(ordered_file):
        print(f"Error: Ordered list file '{ordered_file}' not found!")
        sys.exit(1)

    # Get processing order
    try:
        order = parse_ordered_list(ordered_file)
        print(f"Processing order: {order}")
    except Exception as e:
        print(f"Error reading ordered list: {e}")
        sys.exit(1)

    # Collect all intersection lines
    all_lines = []

    for file_index in order:
        filename = f"all_intersection_lines_{file_index}.txt"
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File '{filepath}' not found, skipping...")
            continue

        print(f"Reading {filename}...")
        lines = parse_intersection_file(filepath)
        print(f"  Found {len(lines)} line segments")
        all_lines.extend(lines)

    if not all_lines:
        print("Error: No intersection lines found!")
        sys.exit(1)

    print(f"Total segments before consolidation: {len(all_lines)}")

    # Consolidate connected segments
    consolidated_lines = consolidate_lines(all_lines)
    print(f"Total segments after consolidation: {len(consolidated_lines)}")

    # Generate G-code
    output_file = os.path.join(output_dir, "parametric_slicer.gcode")
    print(f"Generating G-code to {output_file}...")
    generate_gcode(consolidated_lines, output_file)

    print("G-code conversion completed successfully!")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
