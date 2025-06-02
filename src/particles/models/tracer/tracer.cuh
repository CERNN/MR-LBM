
#ifndef __PARTICLE_MODEL_TRACER_CUH
#define __PARTICLE_MODEL_TRACER_CUH


#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"



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


#endif //__PARTICLE_MODEL_TRACER_CUH