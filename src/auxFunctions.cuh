#ifndef __AUX_FUNCTION_CUH
#define __AUX_FUNCTION_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "var.h"
#include "nodeTypeMap.h"
#include "reduction.cuh"

__host__
void mean_moment( dfloat *fMom,  dfloat *meanMom, int m_index, size_t step);

#endif // !__AUX_FUNCTION_CUH