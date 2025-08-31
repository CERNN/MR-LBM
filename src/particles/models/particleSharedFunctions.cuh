#ifdef PARTICLE_MODEL

#ifndef __PARTICLE_SHARED_FUNCTIONS_CUH
#define __PARTICLE_SHARED_FUNCTIONS_CUH

#include "./../../globalStructs.h"
#include "./../../globalFunctions.h"

// Stencil distance
#if defined(STENCIL_2)
    #define P_DIST 1
#elif defined(STENCIL_4)
    #define P_DIST 2
#else
    #define P_DIST 1
#endif




__device__ __forceinline__  dfloat stencil(dfloat x) {
    dfloat absX = abs(x);
    #if defined STENCIL_2
        if (absX > 1.0) {
            return 0.0;
        }
        else {
            return (1 - x);
        }
    #elif defined STENCIL_4
        if (absX <= 1) {
            return (1.0 / 8.0)*(3.0 - 2.0*absX + sqrt(1 + 4 * absX - 4 * absX*absX));
        }
        else if (absX > 1.0 && absX <= 2.0) {
            return (1.0 / 8.0)*(5.0 - 2.0*absX - sqrt(-7.0 + 12.0*absX - 4.0*absX*absX));
        }
        else {
            return 0.0;
        }
    #endif //STENCIL
    return 0;
}


#endif //__PARTICLE_MODEL_TRACER_CUH
#endif //PARTICLE_MODEL
