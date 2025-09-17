/**
 *  @file mlbm.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief main kernel for moment representation 
 *  @version 0.1.0
 *  @date 01/09/2025
 */

#ifndef __MLBM_H
#define __MLBM_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "var.h"
#include "includeFiles/interface.h"
#include "nodeTypeMap.h"
#ifdef OMEGA_FIELD
    #include "nnf.h"
#endif //OMEGA_FIELD


/**
 *  @brief Updates macroscopics and then performs collision and streaming
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param ghostInterface interface block transfer information
 *  @param d_mean_rho: mean density, used for density correction
 *  @param d_BC_Fx: boundary condition force x
 *  @param d_BC_Fy: boundary condition force x
 *  @param d_BC_Fz: boundary condition force x
 *  @param step: Current time step
 *  @param save: if is necessary save some data
 */
__global__
void gpuMomCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    DENSITY_CORRECTION_PARAMS_DECLARATION(d_)
    BC_FORCES_PARAMS_DECLARATION(d_)
    unsigned int step,
    bool save);

#ifdef LOCAL_FORCES
/**
 * @brief Resets the macroscopic forces to the predefined values FX, FY, FZ
 * @param fMom Pointer to the device array containing the current macroscopic moments.
 */
__global__
void gpuResetMacroForces(dfloat *fMom);
#endif //LOCAL_FORCES

#endif //__MLBM_H