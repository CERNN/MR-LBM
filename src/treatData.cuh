#ifndef __TREAT_DATA_CUH
#define __TREAT_DATA_CUH

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
*   @brief Handles the functions which will do post processing opeations during simulation for faster execution
*   @param h_fMom: host macroscopic field based on block and thread index
*   @param fMom: device macroscopic field based on block and thread index
*   @param fMom_mean: mean flow moments
*   @param nSteps: number of steps of the simulation
*/
__host__
void treatData(
    dfloat* h_fMom,
    dfloat* fMom,
    #if MEAN_FLOW
    dfloat* fMom_mean,
    #endif//MEAN_FLOW
    unsigned int step
);



 /*
*   @brief Calculate the mean of any moment of the flow based  on m_indexthe flow
*   @param fMom: device macroscopic field based on block and thread index
*   @param meanMon: meanMoment for return
*   @param m_index: moment_index
*   @param step: current time step
*   @param target: 0 if meanMom is a device pointer, 1 if is host
*/
__host__
void mean_moment( dfloat *fMom,  dfloat *meanMom, int m_index, size_t step, int target);

/*
*   @brief Calculate the total kinetic energy of the flow
*   @param fMom: device macroscopic field based on block and thread index
*   @param step: current time step
*/
__host__ 
void totalKineticEnergy(dfloat *fMom, size_t step);

#ifdef CONVECTION_DIFFUSION_TRANSPORT
#ifdef CONFORMATION_TENSOR
__host__ 
void totalSpringEnergy(dfloat *fMom,size_t step);
#endif
#endif
/*
*   @brief Calculate the total drag caused by boudary conditions
*   @param d_BC_Fx: device boundary condition force field
*   @param d_BC_Fy: device boundary condition force field
*   @param d_BC_Fz: device boundary condition force field
*   @param step: current time step
*/
__host__ 
void totalBcDrag(dfloat *d_BC_Fx, dfloat* d_BC_Fy, dfloat* d_BC_Fz, size_t step);
 
/*
*   @brief save the velocity profile in the middle of the domian
*   @param fMom: device macroscopic field based on block and thread index
*   @param moment_index: which velocity and direction will be saved
*   @param nSteps: number of steps of the simulation
*/
__host__
void velocityProfile(
    dfloat* fMom,
    int moment_index,
    unsigned int step
);

/*
*   @brief Change field vector order to be used saved in binary
*   @param h_fMom: host macroscopic field based on block and thread index
*   @param omega: omega field if non-Newtonian
*   @param nSteps: number of steps of the simulation
*/
__host__
void probeExport(
    dfloat* fMom, OMEGA_FIELD_PARAMS_DECLARATION unsigned int step
);
__host__
void computeNusseltNumber(
    dfloat* h_fMom,
    dfloat* fMom,
    unsigned int step
);

__host__
void computeTurbulentEnergies(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
);

#endif // !__TREAT_DATA_CUH