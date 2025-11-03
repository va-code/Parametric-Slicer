#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name):

    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
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
    print("Starting Parametric Slicer Pipeline")
    print("=" * 60)
    
    # Define the pipeline sequence
    scripts = [
        "decomposer.py",
        "Adjaceny.py",  # Note: actual filename has typo
        "path.py",
        "Onion3d_IGL.py"
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
        success = run_script(script)
        if not success:
            print(f"\n{'='*60}")
            print(f"Pipeline stopped due to error in {script}")
            print(f"{'='*60}")
            sys.exit(1)
    
    # All scripts completed successfully
    print(f"\n{'='*60}")
    print("✓ Pipeline completed successfully!")
    print("Check the DecompositionOUTPUT folder for results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

