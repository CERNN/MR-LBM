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
#include "nodeTypeMap.h"
#include "./BoundaryConditionsSchemes/D3Q19_MomentBased.cuh"


__device__ void gpuBoundaryConditionPop(
    dim3 threadIdx, dim3 blockIdx, 
    dfloat *pop, dfloat *s_pop,
    char dNodeType);

__device__ void gpuBoundaryConditionMom(    
    dfloat* pop, dfloat &rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat &uzVar, 
    dfloat &pixx  , dfloat &pixy  , dfloat &pixz , 
    dfloat &piyy  , dfloat &piyz  , dfloat &pizz );

__global__ void gpuInitialization_nodeType(
    char *dNodeType);



#endif //__BOUNDARY_CONDITION_CUH