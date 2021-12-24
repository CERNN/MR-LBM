#ifndef __INTERFACE_TRANSFER_CUH
#define __INTERFACE_TRANSFER_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "var.h"


/*
*   @brief Save populations in the correspond interface planes
*   @param threadIdx: thread index
*   @param blockIdx: block index
*   @param pop: populations to be pushed in the interface planes
*   @param fGhostX_0: populations for threadIdx.x == 0
*   @param fGhostX_1: populations for threadIdx.x == NX-1
*   @param fGhostY_0: populations for threadIdx.y == 0
*   @param fGhostY_1: populations for threadIdx.y == NY-1
*   @param fGhostZ_0: populations for threadIdx.z == 0
*   @param fGhostZ_1: populations for threadIdx.z == NZ-1
*/
__device__ void gpuInterfacePush(
    dim3 threadIdx, dim3 blockIdx, dfloat pop[Q],
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1);

/*
*   @brief Pull populations from the neighboor interface planes
*   @param threadIdx: thread index
*   @param blockIdx: block index
*   @param pop: populations to be pushed in the interface planes
*   @param fGhostX_0: populations to be pulled when threadIdx.x == NX-1
*   @param fGhostX_1: populations to be pulled when threadIdx.x == 0
*   @param fGhostY_0: populations to be pulled when threadIdx.y == NY-1
*   @param fGhostY_1: populations to be pulled when threadIdx.y == 0
*   @param fGhostZ_0: populations to be pulled when threadIdx.z == NZ-1
*   @param fGhostZ_1: populations to be pulled when threadIdx.z == 0
*/
__device__ void gpuInterfacePull(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1);

 
#endif // !__LBM_INITIALIZATION_H