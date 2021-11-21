#ifndef __INTERFACE_SPREAD_CUH
#define __INTERFACE_SPREAD_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "var.h"


__device__ void gpuInterfaceSpread(
    dim3 threadIdx, dim3 blockIdx, dfloat pop[Q],
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1);

#endif // !__LBM_INITIALIZATION_H