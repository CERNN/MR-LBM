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

dfloat calcTau_les(dfloat omegaOld, dfloat const auxStressMag, const int step){
    #ifdef MODEL_CONST_SMAGORINSKY
        tau_t = 0.5*sqrt(TAU*TAU+Implicit_const*auxStressMag)-0.5*TAU;
    #endif
    return tau_t;
};

#endif // __LES_H


