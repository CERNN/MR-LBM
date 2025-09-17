/**
 *  @file tracer.cuh
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief handle the simulation for tracer particles
 *  @version 0.1.0
 *  @date 01/09/2025
 */


#ifndef __PARTICLE_MODEL_TRACER_CUH
#define __PARTICLE_MODEL_TRACER_CUH


#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "./../../class/particle.cuh"

#ifdef PARTICLE_MODEL
/**
 *  @brief Perform IBM simulation steps including force interpolation and spreading.
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param streamParticles: cuda stream for particles
 *  @param step: The current simulation time step for collision checking.
 */
__host__
void tracerSimulation(
    ParticlesSoA *pArray,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

/**
 *  @brief Update position of the tracer particles
*
 *  @param particles: particle informaiton
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param step: Current time step
*/
__global__
void tracer_positionUpdate(
    ParticleCenter *particles,
    dfloat *fMom,
    int firstIndex,
    int lastIndex,
    unsigned int step
);

/**
 *  @brief Save the particles position for each time step
 *  @param h_particlePos: host particle position
 *  @param step: Current time step
 */
__host__
void tracer_saveParticleInfo(
    dfloat3 *h_particlePos, 
    unsigned int step
);

#endif //PARTICLE_MODEL

#endif //__PARTICLE_MODEL_TRACER_CUH

