# Parametric-Slicer
A basic implementation of parametric slicing based on convex decomposition

## Quick Start

### Option 1: Using Virtual Environment (Recommended)

First, install python3-venv if you haven't already:
```bash
sudo apt install python3.12-venv
```

Then run the setup script:
```bash
./setup_venv.sh
```

This will create a `.venv` folder, install all dependencies, and set up the environment.

To run the slicer:
```bash
source .venv/bin/activate
python3 main.py
```

### Option 2: Using Nix

```bash
nix-shell shell.nix
python3 main.py
```

## Example Usage

As an example you can decompose the test file "A.stl" which doesn't take a very long time

This will run the complete pipeline (decomposer.py → Adjacency.py → path.py → Onion3d_IGL.py) automatically.

A series of generic 3d printer toolhead positions will be output into the DecompositionOUTPUT folder

Alternatively, you can still run each script individually:
  ~python3 decomposer.py
  ~python3 Adjacency.py
  ~python3 path.py
  ~python3 Onion3d_IGL

A series of generic 3d printer toolhead positions will be output into the DecompositionOUTPUT folder

I would highly recommend changing the decomposer.py file to run the stanford dragon model if you want to see a more complex example. It will take a loooong time to run through some of the steps.

### Manual Package Installation

If you prefer to install packages manually without using the setup script:

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install coacd matplotlib numpy scipy networkx trimesh virtualenv pyglet==1.5
```
      
