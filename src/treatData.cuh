/**
 *  @file treatData.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Data/macroscopics treatment
 *  @version 0.4.0
 *  @date 01/09/2025
 */


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

/**
 *  @brief Handles the functions which will do post processing opeations during simulation for faster execution
 *  @param h_fMom: Pointer to the host array containing the current macroscopic moments.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param fMom_mean: mean flow moments
 *  @param nSteps: number of steps of the simulation
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



/**
 *  @brief Calculate the mean of any moment of the flow based  on m_indexthe flow
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param meanMon: Pointer to the device or host array where the mean moment will be stored.
 *  @param m_index: Index of the moment to be averaged (e.g., M_RHO_INDEX for density).
 *  @param step: Current time step
 *  @param target: 0 if meanMom is a device pointer, 1 if is host
 */
__host__
void mean_moment( dfloat *fMom,  dfloat *meanMom, int m_index, size_t step, int target);

/**
 *  @brief Calculate the total kinetic energy of the flow
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param step: Current time step
 */
__host__ 
void totalKineticEnergy(dfloat *fMom, size_t step);

#ifdef CONVECTION_DIFFUSION_TRANSPORT
#ifdef CONFORMATION_TENSOR
/**
 *  @brief Calculate the total spring energy of the polymers in the flow
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param step: Current time step
 */
__host__ 
void totalSpringEnergy(dfloat *fMom,size_t step);
#endif
#endif
/**
 *  @brief Calculate the turbulent kinetic energy based on the velocity field
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param m_fMom: Pointer to the device array containing the mean flow macroscopic moments.
 *  @param step: Current time step 
 */
__host__ 
void turbulentKineticEnergy(
    dfloat *fMom, 
    dfloat *m_fMom, 
    size_t step
);

/**
 *  @brief Calculate the total drag caused by boudary conditions
 *  @param d_BC_Fx: device boundary condition force field
 *  @param d_BC_Fy: device boundary condition force field
 *  @param d_BC_Fz: device boundary condition force field
 *  @param step: Current time step
 */
__host__ 
void totalBcDrag(dfloat *d_BC_Fx, dfloat* d_BC_Fy, dfloat* d_BC_Fz, size_t step);
 
/**
 *  @brief Save the velocity profile in the middle of the domian
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param moment_index: Which velocity and direction will be saved
 *  @param step: Current time step
 */
__host__
void velocityProfile(
    dfloat* fMom,
    int moment_index,
    unsigned int step
);

/**
 *  @brief Change field vector order to be used saved in binary
 *  @param h_fMom: Pointer to the host array containing the current macroscopic moments.
 *  @param omega: host omega field if non-Newtonian
 *  @param step: Current time step
 */
__host__
void probeExport(
    dfloat* fMom, OMEGA_FIELD_PARAMS_DECLARATION unsigned int step
);

/**
 *  @brief Calculate the Nusselt number based on the temperature field
 *  @param h_fMom: Pointer to the host array containing the current macroscopic moments.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param step: Current time step
 */
__host__
void computeNusseltNumber(
    dfloat* h_fMom,
    dfloat* fMom,
    unsigned int step
);

/**
 *  @brief Calculate the turbulent kinetic energy based on the velocity field
 *  @param h_fMom: Pointer to the host array containing the current macroscopic moments.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param fMom_mean: mean flow moments
 *  @param step: Current time step
 */
__host__
void computeTurbulentEnergies(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
);

#endif // !__TREAT_DATA_CUH