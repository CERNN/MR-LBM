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
#include "includeFiles/interface.h"
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
*   @param d_mean_rho: mean density, used for density correction
*   @param d_BC_Fx: boundary condition force x
*   @param d_BC_Fy: boundary condition force x
*   @param d_BC_Fz: boundary condition force x
*   @param step: current time step
*   @param save: if is necessary save some data
*/
__global__
void gpuMomCollisionStream(
    dfloat *fMom, unsigned int *dNodeType,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1,
    dfloat *gGhostX_0, dfloat *gGhostX_1,
    dfloat *gGhostY_0, dfloat *gGhostY_1,
    dfloat *gGhostZ_0, dfloat *gGhostZ_1,
    #ifdef SECOND_DIST 
    dfloat *g_fGhostX_0, dfloat *g_fGhostX_1,
    dfloat *g_fGhostY_0, dfloat *g_fGhostY_1,
    dfloat *g_fGhostZ_0, dfloat *g_fGhostZ_1,
    dfloat *g_gGhostX_0, dfloat *g_gGhostX_1,
    dfloat *g_gGhostY_0, dfloat *g_gGhostY_1,
    dfloat *g_gGhostZ_0, dfloat *g_gGhostZ_1,
    #endif 
    #ifdef DENSITY_CORRECTION
    dfloat *d_mean_rho,
    #endif
    #ifdef BC_FORCES
    dfloat *d_BC_Fx,dfloat *d_BC_Fy,dfloat *d_BC_Fz,
    #endif 
    unsigned int step,
    bool save);


#endif __MLBM_H