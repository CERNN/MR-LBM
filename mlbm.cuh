#ifndef __MLBM_H
#define __MLBM_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "interfaceSpread.cuh"

__global__
void gpuMomCollisionStream(
    dfloat* fMom, 
    dfloat* fGhostX_0, dfloat* fGhostX_1,
    dfloat* fGhostY_0, dfloat* fGhostY_1,
    dfloat* fGhostZ_0, dfloat* fGhostZ_1
);


#endif __MLBM_H