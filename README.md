# CUDA Gravitybox
Graphical simulation of particles in a gravitational field written using [CUDA API](https://developer.nvidia.com/cuda-zone) for multithreaded processing. Runs smootlhy for up to a million particles.
Tested only under Linux.
## Demo
![demo](https://user-images.githubusercontent.com/73479746/160631052-9542a98e-417f-4e50-8f92-206485df76b2.gif)

Demonstration of different force fields implemented (storm-like, gravity, oscillation).
## Dependencies
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - CUDA API calls
- [GLEW](http://glew.sourceforge.net/) - OpenGL
- [OpenGL Mathematics](https://glm.g-truc.net/) - math functions
- [SDL](www.libsdl.org) - window handling
