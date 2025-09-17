/**
 *  @file les.h
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Infomation about LES
 *  @version 0.1.0
 *  @date 01/09/2025
*/


#ifndef __LES_H
#define __LES_H

#include <math.h>
#include <cmath>
#include "var.h"


#define COMPUTE_SHEAR
#ifdef MODEL_CONST_SMAGORINSKY
constexpr dfloat CONST_SMAGORINSKY = 0.1;
constexpr dfloat INIT_VISC_TURB = 0.0;
constexpr dfloat Implicit_const = 2.0*SQRT_2*3*3/(RHO_0)*CONST_SMAGORINSKY*CONST_SMAGORINSKY;
#endif

/**
 * @brief Calculate the LES relaxation time based on the Smagorinsky model
 * @param omegaOld The old relaxation frequency
 * @param auxStressMag The magnitude of the auxiliary stress tensor
 * @param step The current simulation step
 * @return The calculated relaxation time for LES
 */
dfloat calcTau_les(dfloat omegaOld, dfloat const auxStressMag, const int step){
    #ifdef MODEL_CONST_SMAGORINSKY
        tau_t = 0.5*sqrt(TAU*TAU+Implicit_const*auxStressMag)-0.5*TAU;
    #endif
    return tau_t;
};

#endif // __LES_H


