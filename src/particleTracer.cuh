
#ifndef __PARTICLE_TRACER_CUH
#define __PARTICLE_TRACER_CUH


#include "globalStructs.h"
#include "globalFunctions.h"
#include "interfaceInclude/interface.h"
#include "errorDef.h"
#include "saveData.cuh"
#include <random>


#define NUM_PARTICLES 512

#define STENCIL_4 // Peskin Stencil

// Stencil distance
#if defined(STENCIL_2)
    #define P_DIST 1
#endif

#if defined(STENCIL_4)
    #define P_DIST 2
#endif



/*
*   @brief Evaluate the force distributions based on the stencil
*
*   @param x: the distance between the node thand the reference position
*   @return force weight
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
            return (1.0 / 8.0)*(3.0 - 2.0*absX + sqrt(1 + 4 * absX - 4 * absX*absX));
        }
        else if (absX > 1.0 && absX <= 2.0) {
            return (1.0 / 8.0)*(5.0 - 2.0*absX - sqrt(-7.0 + 12.0*absX - 4.0*absX*absX));
        }
        else {
            return 0.0;
        }
    #endif //STENCIL
}

/*
*   @brief Initialize position of the tracer particles
*
*   @param h_particlePos: host particle position
*   @param d_particlePos: device particle position
*/
__host__
void initializeParticles(    
    dfloat3 *h_particlePos,
    dfloat3 *d_particlePos
);

/*
*   @brief Update position of the tracer particles
*
*   @param d_particlePos: device particle position
*   @param h_particlePos: host particle position
*   @param fMom: macroscopics moments
*   @param streamParticles: CUDA streams for GPUs
*   @param step: current time step
*/
__host__
void updateParticlePos(
    dfloat3 *d_particlePos, 
    dfloat3 *h_particlePos, 
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);


/*
*   @brief Perform the fluid velocity interpolation at the particle location
*
*   @param d_particlePos: device particle position
*   @param fMom: macroscopics moments
*   @param step: current time step
*/
__global__
void velocityInterpolation(
    dfloat3 *d_particlePos, 
    dfloat *fMom,
    unsigned int step
);

/*
*   @brief Save the particles position for each time step
*
*   @param h_particlePos: host particle position
*   @param step: current time step
*/
__host__
void saveParticleInfo(
    dfloat3 *h_particlePos, 
    unsigned int step
);

#endif //__PARTICLE_TRACER_CUH