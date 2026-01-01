#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_script(script_name, debug_mode=False, input_file=None):
    """Run a script with optional debug mode and input file."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    try:
        # Set up environment for debug mode
        env = os.environ.copy()
        if debug_mode:
            env['DEBUG'] = '1'
        if input_file:
            env['INPUT_FILE'] = input_file
        
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            env=env
        )
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Error: {script_name} not found")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error running {script_name}: {e}")
        return False

def main():
    """Run the complete parametric slicer pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Parametric Slicer Pipeline - Process 3D meshes with performance profiling'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed performance profiling'
    )
    parser.add_argument(
        '--profile-output',
        type=str,
        default='DecompositionOUTPUT/Profiling/Profiling_Report.txt',
        help='Output file for profiling report (default: DecompositionOUTPUT/Profiling/Profiling_Report.txt)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='A.stl',
        help='Input STL file to process (default: A.stl)'
    )
    args = parser.parse_args()
    
    print("Starting Parametric Slicer Pipeline")
    print(f"Input file: {args.input}")
    if args.debug:
        print("DEBUG MODE: Performance profiling enabled")
    print("=" * 60)
    
    # Define the pipeline sequence
    scripts = [
        "decomposer.py",
        "Adjacency.py",
        "path.py",
        "Onion3d.py"
    ]
    
    # Check if all scripts exist
    missing_scripts = []
    for script in scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n✗ Error: The following scripts are missing:")
        for script in missing_scripts:
            print(f"  - {script}")
        sys.exit(1)
    
    # Run each script in sequence
    for script in scripts:
        success = run_script(script, debug_mode=args.debug, input_file=args.input)
        if not success:
            print(f"\n{'='*60}")
            print(f"Pipeline stopped due to error in {script}")
            print(f"{'='*60}")
            sys.exit(1)
    
    # All scripts completed successfully
    print(f"\n{'='*60}")
    print("✓ Pipeline completed successfully!")
    print("Check the DecompositionOUTPUT folder for results.")
    
    # If debug mode was enabled, generate final profiling report
    if args.debug:
        print("\nGenerating consolidated profiling report...")
        try:
            # Import and run the report generator
            import subprocess
            result = subprocess.run(
                [sys.executable, "generate_final_report.py", "DecompositionOUTPUT"],
                check=True,
                capture_output=False
            )
            print(f"\n✓ Profiling reports generated in DecompositionOUTPUT/Profiling/")
            print("  See PROFILING_GUIDE.md for help interpreting results.")
        except Exception as e:
            print(f"\n✗ Error generating profiling report: {e}")
        
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

