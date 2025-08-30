#ifndef __PARTICLE_MOVEMENT_H
#define __PARTICLE_MOVEMENT_H

#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "../../class/Particle.cuh"
#include "../../particlesBoundaryCondition.h"

#ifdef PARTICLE_MODEL

__global__
void gpuUpdateParticleOldValues(
    ParticleCenter *particleCenters,
    int firstIndex,
    int lastIndex
);

__global__ 
void gpuUpdateParticleCenterVelocityAndRotation(ParticleCenter particleCenters[NUM_PARTICLES]);

__global__
void gpuParticleMovement(ParticleCenter particleCenters[NUM_PARTICLES]);

#endif //PARTICLE_MODEL

#endif // !__PARTICLE_MOVEMENT_H
