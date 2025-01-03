# MR-LBM for GPU

This repository contains a moment-based implementation of the Lattice Boltzmann Method (LBM), where the collision is performed across moments from 0th to 2nd order. The method is designed for GPU acceleration using CUDA. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation Guide](#installation-guide)
- [Simulation](#simulation)
- [Post Processing](#post-processing)
- [File Structure](#file-structure)
- [Creating a Boundary Case](#creating-a-boundary-case)
- [Using Voxels Immersed Bodies](#using-voxels-immersed-bodies)
- [Benchmark](#benchmark)
- [Gallery](#gallery)
- [Publications](#publications)
- [Updates](#update)
- [License](#license)
- [Contact](#contact)

## Project Overview

The MR-LBM for GPU implements the moment-based Lattice Boltzmann method, where the collision is carried out across moments from the 0th to the 2nd order. This project is intended primarily as a proof of concept, with many features still under development.


This code is heavily based on the LBM implementation from [VISCOPLASTIC-LBM](https://github.com/CERNN/VISCOPLASTIC-LBM) and therefore, shares similar licensing terms.

## Installation Guide

### Requirements

To compile and run the project, you will need:

- C++ compiler (e.g., MSVC)
- Nvidia GPU drivers
- CUDA toolkit

The CUDA toolkit, provided by Nvidia, includes both the GPU drivers and the necessary CUDA libraries. Ensure that your GPU supports compute capability 3.5 or higher. Note that the software is designed to run on a single GPU, and multi-GPU setups are not currently supported.

### Compilation

To compile the project, you can use the provided [bash script](./src/compile.sh). The script includes instructions on how to modify it based on your GPU’s compute capability and the necessary arguments. 

### Installation Steps
1. Install the CUDA toolkit from [Nvidia's website](https://developer.nvidia.com/cuda-toolkit).
2. Modify the `compile.sh` file if necessary for your GPU.
3. Run the script to compile the project.

## Simulation

### Output Files

The program generates binary output files containing macroscopic quantities such as density and velocity, as well as an information file that describes the simulation parameters (lattice size, tau, velocity set, etc.).

To convert these binary files into a format that can be interpreted by other software (e.g., Paraview), a Python post-processing tool is provided. 

## Post Processing

To process the output binary files, you will need Python 3 and the following dependencies:

- glob
- numpy
- os
- pyevtk
- matplotlib
To convert the binary files to `.vtr` files (used in Paraview), use the following command:

```bash
python exampleVtk.py "PATH_FILES/ID_SIM"
```

## File Structure

| No  | File Name              | Details                                                |
| --- | ---------------------- | ------------------------------------------------------ |
| 1   | main                   | Main application code                                  |
| 2   | mlbm                   | Core kernel with streaming-collision operations        |
| 3   | var                    | Simulation parameters                                  |
| 4   | compile.sh             | Compilation script (edit for correct CUDA version)     |
| 5   | definitions.h          | Constants used within LBM                              |
| 6   | arrayIndex.h           | Index calculation for moments                          |
| 7   | auxFunctions.h         | Auxiliary functions                                    |
| 8   | cases                  | Folder for different simulation cases                  |
| 9   | cases/bc_definition    | Boundary condition definitions for lattices            |
| 10  | cases/bc_initialization| Boundary condition flags                               |
| 11  | cases/constants        | Constants specific to the case                         |
| 12  | cases/flow_initialization | Flow initialization parameters                      |
| 13  | cases/model            | Model parameters (e.g., velocity set, collision model) |
| 14  | cases/output           | Data export parameters                                 |
| 15  | colrec                 | Collision and reconstruction files for moments         |
| 16  | includeFiles/popSave   | Load population from global memory                     |
| 17  | includeFiles/popLoad   | Save population to global memory                       |
| 18  | includeFiles/interface | Definition of boundary conditions (wall or periodic)   |
| 19  | checkpoint             | Functions for generating simulation checkpoints        |
| 20  | errorDef               | Error handling functions                               |
| 21  | globalFunctions        | Index functions                                        |
| 22  | globalStructs          | Structures for device and host                         |
| 23  | lbmInitialization      | Field initialization functions                         |
| 24  | nnf                    | Non-Newtonian fluid definitions                        |
| 25  | nodeTypeMap            | Boundary condition node type map                       |
| 26  | particleTracer         | Functions for particle tracing                         |
| 27  | reduction              | Parallel reduction functions                           |
| 28  | saveData               | Functions to save simulation data                      |

## Creating a Boundary Case

To define a new boundary case, you need to create files within the `cases` folder. Each case should contain the following files:

1. `bc_definition`: Defines the mathematical equations for the moments (0th to 2nd order).
2. `bc_initialization`: Defines the boundary condition flags.
3. `constants`: Defines simulation parameters (mesh size, velocity, etc.).
4. `flow_initialization`: Defines how the flow is initialized.
5. `model`: Defines model parameters (e.g., velocity set, collision model).
6. `output`: Defines output parameters for data export.

Additional modelels like thermal or viscoelastic will require additional files.

## Using Voxels Immersed Bodies

To use voxels for immersed bodies, follow these steps:

1. Create a CSV file containing the coordinates of the solid nodes.
2. Add an `include` directive for `VOXEL_FILENAME` in the `constants` file.
3. Include `VOXEL_BC_DEFINE` in the `bc_definition`.

## Benchmark

The following table summarizes performance benchmarks on various GPUs:

| GPu           | sm | GPU Clock (GHz) | Memory Clock (GHz) | Block Size      | MLUPs     | Observations                          |
|---------------|----|-----------------|--------------------|-----------------|-----------|---------------------------------------|
|  RTX 4090 OC  | 89 |  3.0 GHz        |  1.5 GHz           |  8x8x8          |  9075     |                                       |
|  RTX 4090     | 89 |  2.8 GHz        |  1.3 GHz           |  8x8x8          |  7899     |                                       |
|  RTX 4060     | 89 |  2.8 GHz        |  2.1 GHz           |  8x8x8          |  2167     |                                       |
|  RTX 4060     | 89 |  2.8 GHz        |  2.1 GHz           |  8x8x4          |  1932     |                                       |
|---------------|----|-----------------|--------------------|-----------------|-----------|---------------------------------------|
|  RTX 3060 OC  | 86 |  2.0 GHz        |  2.0 GHz           |  8x8x8          |  3083     |                                       |
|  RTX 3060     | 86 |  1.8 GHz        |  1.8 GHz           |  8x8x8          |  2755     |                                       |
|---------------|----|-----------------|--------------------|-----------------|-----------|---------------------------------------|
| A100          | 80 | ### GHz         | ### GHz            | 16x16x8 (D)     | #####     | waiting to test                       |
| A100          | 80 | ### GHz         | ### GHz            | 16x8x8 (D)      | #####     | waiting to test                       |
| A100          | 80 | ### GHz         | ### GHz            | 8x8x8           | #####     | waiting to test                       |
|---------------|----|-----------------|--------------------|-----------------|-----------|---------------------------------------|
|  RTX 2060     | 75 |  1.9   GHz      |  1.7 GHz           |  8x8x8          |  2357     |                                       |
|  GTX 1660     | 75 |  1.9 GHz        |  2.0 GHz           |  8x8x8          |  1252     |                                       |
|  GTX 1660     | 75 |  1.9 GHz        |  2.0 GHz           |  8x8x4          |  1251     |                                       |
|  GTX 1660     | 75 |  1.9 GHz        |  2.0 GHz           |  16x4x4         |  1212     |                                       |
|---------------|----|-----------------|--------------------|-----------------|-----------|---------------------------------------|
|  K20x         | 35 |  0.7 GHz        |  1.3 GHz           |  8x8x4          |  730      | Limited by GPU, Memory controler 47%  |
|  K20x         | 35 |  0.7 GHz        |  1.3 GHz           |  8x8x8          |  670      | Limited by GPU, Memory controler 40%  |
|  K80          | 35 |  0.8 GHz        |  1.2 GHz           |  8x8x8          | 1142      | Using 1 Chip                          |
|---------------|----|-----------------|--------------------|-----------------|-----------|---------------------------------------|

Currently tested speed using benchmark case FP32, differences may occur due to frequencies and thermal throttle.

### Note:
- **D** refers to dynamic allocation of shared memory, which is required to increase the maximum shared memory per block to 48 KB.

## Gallery

*In progress*

## Publications

- [DOI: 10.1063/5.0209802](https://doi.org/10.1063/5.0209802)
- [DOI: 10.1016/j.jnnfm.2024.105198](https://doi.org/10.1016/j.jnnfm.2024.105198)
- [ResearchGate: Evaluating the Impact of Boundary Conditions on MR-LBM](https://www.researchgate.net/publication/378070516_Evaluating_the_Impact_of_Boundary_Conditions_on_the_MR-LBM)
- [DOI: 10.1016/j.jnnfm.2023.105030](https://doi.org/10.1016/j.jnnfm.2023.105030)
- [DOI: 10.1002/fld.5185](https://doi.org/10.1002/fld.5185)

## TODO

The [TODO.todo](./TODO.todo) file contains list of other objectives that may or not may be implemented. Either contain items to improve performance or enchance the capabilities of the model.


## License

This software is provided under the [GPLv2 license](./LICENSE.txt).

## Contact

For bug reports or issues, please use the GitHub issue tracker for better visibility. You can also contact the maintainers via email at:

- _marcoferrari@alunos.utfpr.edu.br_
