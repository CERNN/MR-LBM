#ifndef __PARTICLE_MODEL_IBM_CUH
#define __PARTICLE_MODEL_IBM_CUH

#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "../../class/Particle.cuh"
#include "../../particlesBoundaryCondition.h"

#include "ibmMacrsAux.cuh"
#include "../dem/particleMovement.cuh"
#include "../dem/collisionDetection.cuh"


void ibmSimulation(
    ParticlesSoA particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);


#endif



