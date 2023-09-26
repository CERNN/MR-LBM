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


/*
*   @brief Calculate the mean of any moment of the flow based  on m_indexthe flow
*   @param fMom: device macroscopic field based on block and thread index
*   @param meanMon: meanMoment for return
*   @param m_index: moment_index
*   @param step: current time step
*/
__host__
void mean_moment( dfloat *fMom,  dfloat *meanMom, int m_index, size_t step);


/*
*   @brief Calculate the total kinetic energy of the flow
*   @param fMom: device macroscopic field based on block and thread index
*   @param SEK: sum of kinetic energy
*   @param step: current time step
*/
__host__ 
void totalKineticEnergy(dfloat *fMom, dfloat *SEK, size_t step);


#endif // !__AUX_FUNCTION_CUH