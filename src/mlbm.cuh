#ifndef __MLBM_H
#define __MLBM_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "var.h"
#include "interfaceInclude/interface.h"
#include "boundaryCondition.cuh"
#ifdef NON_NEWTONIAN_FLUID
    #include "nnf.h"
#endif


/*
*   @brief Updates macroscopics and then performs collision and streaming
*   @param fMom: macroscopics moments
*   @param fGhostX_0: populations to be pulled when threadIdx.x == NX-1
*   @param fGhostX_1: populations to be pulled when threadIdx.x == 0
*   @param fGhostY_0: populations to be pulled when threadIdx.y == NY-1
*   @param fGhostY_1: populations to be pulled when threadIdx.y == 0
*   @param fGhostZ_0: populations to be pulled when threadIdx.z == NZ-1
*   @param fGhostZ_1: populations to be pulled when threadIdx.z == 0
*   @param gGhostX_0: populations to be saved when threadIdx.x == 0
*   @param gGhostX_1: populations to be saved when threadIdx.x == NX-1
*   @param gGhostY_0: populations to be saved when threadIdx.y == 0
*   @param gGhostY_1: populations to be saved when threadIdx.y == NY-1
*   @param gGhostZ_0: populations to be saved when threadIdx.z == 0
*   @param gGhostZ_1: populations to be saved when threadIdx.z == NZ-1
*/
__global__
void gpuMomCollisionStream(
    dfloat *fMom, unsigned char *dNodeType,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1,
    dfloat *gGhostX_0, dfloat *gGhostX_1,
    dfloat *gGhostY_0, dfloat *gGhostY_1,
    dfloat *gGhostZ_0, dfloat *gGhostZ_1,
    #ifdef DENSITY_CORRECTION
    dfloat *d_mean_rho,
    #endif
    #ifdef LOCAL_FORCES
    dfloat *d_L_Fx,dfloat *d_L_Fy,dfloat *d_L_Fz,
    #endif 
    unsigned int step);


#endif __MLBM_H