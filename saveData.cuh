#ifndef __SAVE_DATA_H
#define __SAVE_DATA_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"


__host__
void linearMacr(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz
);


__host__
void saveMacr(
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    unsigned int nSteps
);

void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append
);

std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext
);

void folderSetup();



#endif __SAVE_DATA_H