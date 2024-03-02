# MR-LBM for GPU

This is a moment represent LBM (Lattice Boltzmann method), where the collision is performed in momentos from 0th to 2nd order.

The theory can be found in the article: https://doi.org/10.1002/fld.5185
Which should be used as software citation.

It currently does not support many features, and only was created to be used as proof-of-concept.

Great part of the files share the same code as https://github.com/CERNN/VISCOPLASTIC-LBM, and therefore it will share similar licence.

## Compilation

The requirements are:
* C++ compiler (MSVC for exemple)
* Nvidia drivers must be installed
* CUDA API must be installed

Both can be obtained in "CUDA Toolkit", provided by Nvidia.

The code supports Nvidia's GPUs with compute capability 3.5 or higher. The program runs in only one GPU, multi-GPU is not supported yet.

For compilation, a [bash file](./src/compile.sh) is provided. It contains the commands used to compile and the instructions for altering it according to the GPU compute capability and the arguments to pass to it.


## Simulation

The output of the simulations are binary files with the content of macroscopics (density, velocity, etc.), an information file with the simulation parameters (lattice size, tau, velocity set, etc.). To convert from binary to interpretable data, a Python application is provided. "Post Processing" gives more details on that.


## Post Processing

Since the program exports macroscopics in binary format, it is necessary to process it. For that, Python source files are provided. _python3_ is required and the packages dependecies are:
* glob
* numpy
* os
* pyevtk
* matplotlib

to convert the bin files into .vtr to be used on paraview, in the post folder use: python exampleVtk.py "PATH_FILES/ID_SIM"
Current setup is for the flow around a sphere

## File Structure
| No | File Name | Details 
|----|------------|-------|
| 1  | main | main
| 2  | mlbm | core kernel with streaming-collision operations
| 3  | var | Simulation parameters
| 4  | compile.sh | compile shell script, edit for the correct cuda version
|----|------------|-------|
| 5  | definitions.h | constants used within LBM
| 6  | arrayIndex.h | index calculation for moments
| 7  | auxFunctions.h | auxialiary functions that may be used during computation
| 8  | boundary conditions | definition of boundary condition for lattices
| 9  | BoundaryConditions/Boundary_initialization_files | definition of boundary condition for lattices for each case 
| 10 | BoundaryConditions/IncludeMlbmBc_MOM | moment based boundary conditions for each case
| 11 | BoundaryConditions/IncludeMlbmBc_POP | population based boundary conditions for each case
| 12 | interfaceInclude/popSave | load population from global memory
| 13 | interfaceInclude/popLoad | save population into global memory
| 14 | interfaceInclude/interface | definition of the frontier if is wall or periodic for each case
| 15 | checkpoint | functions to generation simulation checkpoint
| 16 | errorDef | error definition functions
| 17 | globalFunctions | index functions
| 18 | globalStructs | structs for device and host
| 19 | lbmInitialization | field initialization functions
| 20  | nnf | non-Newtonian fluid definitions
| 21  | nodeTypeMap | boundary conditions node type map defintions
| 22  | particleTracer | particle tracer functions
| 23  | reduction | parallel reduction functions used for sums over the domain
| 24  | saveData | functions to save simulation data


## Creating a boundary case
In order to create a NEW_CASE is necessary to create modify three files:

1. Boundary_initialization_files/NEW_CASE : should contain the binary value of which type of boundary condition is applied to each cell. This file will be loaded during the initilazation process. As default fluid nodes without BC should be 0b0000000 and solid nodes 0b11111111. Currently support up to 254 different boundary conditions for each case

2. IncludeMlbmBc_###/NEW_CASE : Either moment or population based. This file should contain how the boundary condition is calculated for each case.

3. interfaceInclude/interface : The definition if the frontier in each of the three directions (x,y, and z) will be periodic, otherwise has be defined as a WALL. Necessary to avoid population leakeage between frontiers.

## using voxels immersed bodies
1.  create a csv with the coordinates values for solid nodes
2.  edit VOXEL_FILENAME defintion in var.h
3.  externalFlow create bc creates a fixed inlet velocity and outflow in the z-direction


## Gallery

## Publications
https://doi.org/10.1016/j.jnnfm.2024.105198
https://www.researchgate.net/publication/378070516_Evaluating_the_Impact_of_Boundary_Conditions_on_the_MR-LBM
https://doi.org/10.1016/j.jnnfm.2023.105030
https://doi.org/10.1002/fld.5185

## Update
Currently 0 commits behind local development version.


## License

This software is provided under the [GPLv2 license](./LICENSE.txt).

## Contact

For bug report or issue adressing, usage of git resources (issues/pull request) is encouraged. Contact via email: _marcoferrari@alunos.utfpr.edu.br_ and/or _cernn-ct@utfpr.edu.br_.