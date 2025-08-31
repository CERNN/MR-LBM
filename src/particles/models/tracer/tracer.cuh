#ifdef PARTICLE_MODEL
#ifndef __PARTICLE_MODEL_TRACER_CUH
#define __PARTICLE_MODEL_TRACER_CUH


#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "./../../class/particle.cuh"

/*
*   @brief Handle all simulation process of the tracer particles
*
*   @param particles: particle informaiton
*   @param fMom: macroscopics moments
*   @param streamParticles: CUDA streams for GPUs
*   @param step: current time step
*/
__host__
void tracerSimulation(
    ParticlesSoA *pArray,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

/*
*   @brief Update position of the tracer particles
*
*   @param particles: particle informaiton
*   @param fMom: macroscopics moments
*   @param step: current time step
*/
__global__
void tracer_positionUpdate(
    ParticleCenter *particles,
    dfloat *fMom,
    int firstIndex,
    int lastIndex,
    unsigned int step
);

/*
*   @brief Save the particles position for each time step
*
*   @param h_particlePos: host particle position
*   @param step: current time step
*/
__host__
void tracer_saveParticleInfo(
    dfloat3 *h_particlePos, 
    unsigned int step
);


#endif //__PARTICLE_MODEL_TRACER_CUH

#endif //PARTICLE_MODEL
