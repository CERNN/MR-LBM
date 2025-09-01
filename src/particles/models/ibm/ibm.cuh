/**
*   @file ibm.cuh
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @author Ricardo de Souza
*   @brief IBM steps: perform interpolation and spread force
*   @version 0.4.0
*   @date 01/09/2025
*/


#ifndef __PARTICLE_MODEL_IBM_CUH
#define __PARTICLE_MODEL_IBM_CUH

#include "../../../globalStructs.h"
#include "../../../globalFunctions.h"
#include "../particleSharedFunctions.cuh"
#include "../../../includeFiles/interface.h"
#include "../../../errorDef.h"
#include "../../../saveData.cuh"
#include "../../class/Particle.cuh"

#include "../dem/particleMovement.cuh"
#include "../dem/collision/collisionDetection.cuh"

#ifdef PARTICLE_MODEL
void ibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

__global__ 
void gpuResetNodesForces(
    IbmNodesSoA* particlesNodes,
    unsigned int step
);


__global__
void gpuForceInterpolationSpread(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    dfloat *fMom,
    unsigned int step
);

__global__
void gpuParticleNodeMovement(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif



