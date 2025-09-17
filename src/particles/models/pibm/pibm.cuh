#ifndef __PARTICLE_MODEL_PIBM_CUH
#define __PARTICLE_MODEL_PIBM_CUH


#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "./../../class/Particle.cuh"

#ifdef PARTICLE_MODEL

/**
 *  @brief Perform PIBM simulation steps including force interpolation and spreading.
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param streamParticles: cuda stream for particles
 *  @param step: The current simulation time step for collision checking.
 */
void pibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif //__PARTICLE_MODEL_PIBM_CUH


