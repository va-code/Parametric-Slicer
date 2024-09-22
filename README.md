# Parametric-Slicer
A basic implementation of parametric slicing based on convex decomposition

As an example you can decompose the test file "A.stl" which doesn't take a very long time:

~nix-shell shell.nix

  ~python3 decomposer.py
  
  ~python3 Adjacency.py
  
  ~python3 path.py
  
  ~python3 Onion3d_IGL

A series of generic toolhead positions will be output into the DecompositionOUTPUT folder

I would highly recommend changing the decomposer.py file to run the stanford dragon model if you want to see a more complex example. It will take a loooong time to run through some of the steps.
