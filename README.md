# LBM Based on moments for GPU

This is a LBM (Lattice Boltzmann method) based on moments, where the collision is performed in momentos from 0th to 2nd order.

The theory of the algorithm can be found in the article: §§§§§§
Which should be used as software citation.

It currently does not support many features, and only was created to be used as proof-of-concept.

Great part of the files share the same code as https://github.com/CERNN/VISCOPLASTIC-LBM, and therefore it will share same kind of licence.

## Compilation

The requirements are:
* Nvidia drivers must be installed
* CUDA API must be installed

Both can be obtained in "CUDA Toolkit", provided by Nvidia.

The code supports Nvidia's GPUs with compute capability 3.5 or higher. The program runs in only one GPU, multi-GPU support is in development.

For compilation, a [bash file](./compile.sh) is provided. It contains the commands used to compile and the instructions for altering it according to the GPU compute capability and the arguments to pass to it.


## Simulation

The output of the simulations are binary files with the content of macroscopics (density, velocity, etc.), an information file with the simulation parameters (lattice size, tau, velocity set, etc.), It is also provided an option to output treated data, values obtained by the treatment of simulation macroscopics, as residual and average density. The simulation information and the treated data are also printed on screen. To convert from binary to interpretable data, a Python application is provided. "Post Processing" gives more details on that.


## Post Processing

Since the program exports macroscopics in binary format, it is necessary to process it. For that, Python source files are provided. _python3_ is required and the packages dependecies are:
* glob
* numpy
* os
* pyevtk
* matplotlib

## License

This software is provided under the [GPLv2 license](./LICENSE.txt).

## Contact

For bug report or issue adressing, usage of git resources (issues/pull request) is encouraged. Contact via email: _marcoferrari@alunos.utfpr.edu.br_ and/or _cernn-ct@utfpr.edu.br_.