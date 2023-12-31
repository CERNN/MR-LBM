# MR-LBM for GPU

This is a moment represent LBM (Lattice Boltzmann method), where the collision is performed in momentos from 0th to 2nd order.

The theory can be found in the article: https://doi.org/10.1002/fld.5185
Which should be used as software citation.

It currently does not support many features, and only was created to be used as proof-of-concept.

Great part of the files share the same code as https://github.com/CERNN/VISCOPLASTIC-LBM, and therefore it will share similar licence.

## Compilation

The requirements are:
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


## File Structure
| No | File Name | Details 
|----|------------|-------|
| 1  | main | main
| 2  | mlbm | core kernel with streaming-collision operations
| 3  | var | Simulation parameters
| 4  | saveData | functions to save simulation data
| 5  | reduction | parallel reduction functions
| 6  | particleTracer | particle tracer functions
| 7  | nnf | non-Newtonian fluid definitions
| 8  | lbmInitialization | field initialization functions
| 9  | globalStructs | structs for device and host
| 10  | globalFunctions | index functions
| 11  | errorDef | error definition functions
| 12  | checkpoint | functions to generation simulation checkpoint
| 13  | auxFunctions | auxiliary functions to treat simulation data
| 14  | interfaceInclude/popSave | load population from global memory
| 15  | interfaceInclude/popLoad | save population into global memory
| 16  | interfaceInclude/interface | definition of the frontier if is wall or periodic for each case
| 17  | boundary conditions | definition of boundary condition for lattices
| 18  | BoundaryConditions/Boundary_initialization_files | definition of boundary condition for lattices for each case 
| 19  | BoundaryConditions/IncludeMlbmBc_MOM | moment based boundary conditions for each case
| 20  | BoundaryConditions/IncludeMlbmBc_POP | population based boundary conditions for each case


## Creating a case
In order to create a NEW_CASE is necessary to create modify three files:

1. Boundary_initialization_files/NEW_CASE : should contain the binary value of which type of boundary condition is applied to each cell. This file will be loaded during the initilazation process. As default fluid nodes without BC should be 0b0000000 and solid nodes 0b11111111. Currently support up to 254 different boundary conditions for each case

2. IncludeMlbmBc_###/NEW_CASE : Either moment or population based. This file should contain how the boundary condition is calculated for each case.

3. interfaceInclude/interface : The definition if the frontier in each of the three directions (x,y, and z) will be periodic, otherwise has be defined as a WALL. Necessary to avoid population leakeage between frontiers.

## Gallery

## Publications

## Update
Currently 14 commits behind local development version.


## License

This software is provided under the [GPLv2 license](./LICENSE.txt).

## Contact

For bug report or issue adressing, usage of git resources (issues/pull request) is encouraged. Contact via email: _marcoferrari@alunos.utfpr.edu.br_ and/or _cernn-ct@utfpr.edu.br_.