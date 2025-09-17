/**
 *  @file particleSharedFunctions.cuh
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief shared functions for particle simulation
 *  @version 0.1.0
 *  @date 01/09/2025
 */

#ifndef __PARTICLE_SHARED_FUNCTIONS_CUH
#define __PARTICLE_SHARED_FUNCTIONS_CUH

#include "./../../globalStructs.h"
#include "./../../globalFunctions.h"

#ifdef PARTICLE_MODEL

// Stencil distance
#if defined(STENCIL_2)
    #define P_DIST 1
#elif defined(STENCIL_4)
    #define P_DIST 2
#elif defined(STENCIL_COS)
    #define P_DIST 2
#else
    #define P_DIST 1
#endif



/**
 *  @brief Compute the value of the interpolation stencil function based on the distance x.
 *  @param x The distance from the point of interest.
 *  @return The value of the stencil function at distance x.
 */
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
            return (3.0 - 2.0*absX + sqrt(1 + 4 * absX - 4 * absX*absX))/8;
        }
        else if (absX > 1.0 && absX <= 2.0) {
            return (5.0 - 2.0*absX - sqrt(-7.0 + 12.0*absX - 4.0*absX*absX))/8;
        }
        else {
            return 0.0;
        }
    #elif defined STENCIL_COS
        if (absX <= 2){
            return (cos(M_PI*absX/2)+1.0)/4;
        }   
        else{
            return 0.0;
        }
    #endif //STENCIL
    return 0;
}

#endif //PARTICLE_MODEL
#endif //__PARTICLE_MODEL_TRACER_CUH

