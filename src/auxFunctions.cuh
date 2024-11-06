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
#include "saveData.cuh"

#include <fstream>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision


/*
*   @brief Calculate the mean density value of the field and save it
*   @param fMom: device macroscopic field based on block and thread index
*   @param step: current time step
*/
__host__ 
void mean_rho(dfloat *fMom, size_t step, dfloat *d_mean_rho);

__host__
void meanFlowComputation(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
);


#endif // !__AUX_FUNCTION_CUH