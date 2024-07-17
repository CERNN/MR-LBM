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
| 7  | auxFunctions.h | auxiliary functions that may be used during computation
| 8  | cases | folder containing the each type of problem
| 9  | cases/bc_definition | definition of boundary condition for lattices for each case 
| 10 | cases/bc_initilization | flags for the boundary condition
| 11 | cases/constants | constants of the case
| 12 | cases/flow_initialization | how the flow is initialized
| 13 | colrec | collision and reconstruction files for the moments (only MR-LBM, others not fully implemented yet)
| 14 | includeFiles/popSave | load population from global memory
| 15 | includeFiles/popLoad | save population into global memory
| 16 | includeFiles/interface | definition of the frontier if is wall or periodic for each case
| 17 | checkpoint | functions to generation simulation checkpoint
| 18 | errorDef | error definition functions
| 19 | globalFunctions | index functions
| 20 | globalStructs | structs for device and host
| 21 | lbmInitialization | field initialization functions
| 22 | nnf | non-Newtonian fluid definitions
| 23 | nodeTypeMap | boundary conditions node type map defintions
| 24 | particleTracer | particle tracer functions
| 25 | reduction | parallel reduction functions used for sums over the domain
| 26 | saveData | functions to save simulation data


## Creating a boundary case
Cases are managed in the cases folder, where each case has to have to following files:

1. bc_definition: Defines the mathematical equations to compute the moments of 0th to 2nd order.
2. bc_initialization: Defines the boundary condition flag
3. constants: define the simulation parameters, ie, mesh size, velocity and so on.
4. flow initialization: define how the flow will be initialized.


## using voxels immersed bodies
1. create a csv with the coordinates values for solid nodes
2. add an include with VOXEL_FILENAME defintion in constants
3. add an incluence for VOXEL_BC_DEFINE in the bc_definition

## Benchmark

Current tested speed Benchmark values using FP32

| GPU | sm |GPU Clock (GHz) | Memory Clock (GHz) |  Block Size | MLUPS | Observations 
|-------------|----|---------|----------|-------------|-------|
| RTX 4090    | 89 | ### GHz | ### GHz  | 16x8x8 (D)  | ##### | waiting availability to test
| RTX 4090 OC | 89 | 3.0 GHz | 1.5 GHz  | 8x8x8       | 9075  |
| RTX 4090    | 89 | 2.8 GHz | 1.3 GHz  | 8x8x8       | 7899  | 
| RTX 4060    | 89 | 2.8 GHz | 2.1 GHz  | 8x8x8       | 2167  |
| RTX 4060    | 89 | 2.8 GHz | 2.1 GHz  | 8x8x4       | 1932  |
|-------------|----|---------|----------|-------------|-------|
| RTX 3060 OC | 86 | 2.0 GHz | 2.0 GHz  | 8x8x8       | 3083  |
| RTX 3060    | 86 | 1.8 GHz | 1.8 GHz  | 8x8x8       | 2755  |
|-------------|----|---------|----------|-------------|-------|
| A100        | 80 | ### GHz | ### GHz  | 16x16x8 (D) | ##### | waiting to test
| A100        | 80 | ### GHz | ### GHz  | 16x8x8 (D)  | ##### | waiting to test
| A100        | 80 | ### GHz | ### GHz  | 8x8x8       | ##### | waiting to test
|-------------|----|---------|----------|-------------|-------|
| RTX 2060    | 75 | 1.9 GHz | 1.7 GHz  | 8x8x8       | 2357  |
| GTX 1660    | 75 | 1.9 GHz | 2.0 GHz  | 8x8x8       | 1252  |
| GTX 1660    | 75 | 1.9 GHz | 2.0 GHz  | 8x8x4       | 1251  |
| GTX 1660    | 75 | 1.9 GHz | 2.0 GHz  | 16x4x4      | 1212  |
|-------------|----|---------|----------|-------------|-------|
| K20x        | 35 | 0.7 GHz | 1.3 GHz  | 8x8x4       | 730   | Limited by GPU, Memory controler 47%
| K20x        | 35 | 0.7 GHz | 1.3 GHz  | 8x8x8       | 670   | Limited by GPU, Memory controler 40%
| K80         | 35 | ### GHz | ### GHz  | 8x8x8       | ##### | waiting to test
|-------------|----|---------|----------|-------------|-------|

D correspond to dynamic allocation of shared memory, which is required to increase the maximum shared memory per block of 48 KB

## Gallery

## Publications

https://doi.org/10.1063/5.0209802

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
