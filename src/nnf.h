#ifndef __NNF_H
#define __NNF_H

#include <math.h>
#include <cmath>
#include "var.h"


/* ------------------------ NON NEWTONIAN FLUID TYPE ------------------------ */
#ifdef BINGHAM
// Inputs
constexpr dfloat Bn = 5.0;

constexpr dfloat S_Y = Bn * VISC * U_MAX / L;                // Yield stress 0.00579
// Calculated variables
constexpr dfloat OMEGA_P = 1 / (3.0*VISC+0.5);    // 1/tau_p = 1/(3*eta_p+0.5)
#endif
/* -------------------------------------------------------------------------- */




#ifdef NON_NEWTONIAN_FLUID
    __device__ 
    dfloat __forceinline__ calcOmega(dfloat omegaOld, dfloat const auxStressMag){
        return OMEGA_P * myMax(0.0, (1 - S_Y / auxStressMag));
    }
#endif // NON_NEWTONIAN_FLUID


#endif // __NNF_H