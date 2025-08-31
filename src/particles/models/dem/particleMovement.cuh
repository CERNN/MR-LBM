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
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,    
    unsigned int step
);

__global__ 
void gpuUpdateParticleCenterVelocityAndRotation(
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,    
    unsigned int step
);

__global__
void gpuParticleMovement(
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,    
    unsigned int step
);

#endif //PARTICLE_MODEL

#endif // !__PARTICLE_MOVEMENT_H
