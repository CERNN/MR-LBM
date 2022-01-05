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


__device__ void gpuBoundaryConditionPop(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop, dfloat *s_pop);

__device__ void gpuBoundaryConditionMom(    
    dim3 threadIdx, dim3 blockIdx , dfloat* pop, dfloat &rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat &uzVar, 
    dfloat &pixx  , dfloat &pixy  , dfloat &pixz , 
    dfloat &piyy  , dfloat &piyz  , dfloat &pizz );

__global__ void gpuInitialization_nodeType(
    char *dNodeType);



#endif //__BOUNDARY_CONDITION_CUH