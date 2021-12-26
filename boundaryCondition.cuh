#ifndef __BOUNDARY_CONDITION_CUH
#define __BOUNDARY_CONDITION_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "var.h"


__device__ void gpuBoundaryCondition(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop, dfloat *s_pop)

#endif //__BOUNDARY_CONDITION_CUH