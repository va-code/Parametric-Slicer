let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    buildInputs = [
      pkgs.python3
      pkgs.python3Packages.matplotlib
      pkgs.python3Packages.numpy
      pkgs.python3Packages.scipy
      pkgs.python3Packages.networkx
      pkgs.python3Packages.trimesh
      pkgs.cgal
      pkgs.python3Packages.virtualenv
    ];

    shellHook = ''
      # Create a virtual environment if it doesn't exist
      if [ ! -d ".venv" ]; then
        virtualenv .venv
      fi

      # Activate the virtual environment
      source .venv/bin/activate

      # Install additional packages using pip
      pip install bpy libigl coacd 
      pip install pyglet==1.5
    '';
  }

