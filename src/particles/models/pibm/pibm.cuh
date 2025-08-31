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




void pibmSimulation(
    ParticlesSoA particles,
    cudaStream_t streamParticles,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif


