/**
*   @file boundaryCondition.cuh
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @brief Definitions for boundary conditions
*   @version 0.1.0
*   @date 01/09/2025
*/

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

__device__ 
void gpuBoundaryConditionMom(    
    dfloat* pop, dfloat &rhoVar, unsigned int dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat &uzVar, 
    dfloat &pixx  , dfloat &pixy  , dfloat &pixz , 
    dfloat &piyy  , dfloat &piyz  , dfloat &pizz 
);

__global__ 
void gpuInitialization_nodeType(
    unsigned int *dNodeType
);



#endif //__BOUNDARY_CONDITION_CUH