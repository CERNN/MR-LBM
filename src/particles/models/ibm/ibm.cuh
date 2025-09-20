/**
 *  @file ibm.cuh
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief IBM steps: perform interpolation and spread force
 *  @version 0.4.0
 *  @date 01/09/2025
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


/**
 *  @brief Perform IBM simulation steps including force interpolation and spreading.
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param streamParticles: cuda stream for particles
 *  @param step: The current simulation time step for collision checking.
 */
void ibmSimulation(
    ParticlesSoA* particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
);

/**
 *  @brief Reset the forces on IBM nodes to zero.
 *  @param particlesNodes: Pointer to the IbmNodesSoA structure containing IBM node data.
 *  @param step: The current simulation time step for collision checking.
 */
__global__ 
void ibmResetNodesForces(
    IbmNodesSoA* particlesNodes,
    unsigned int step
);


/**
 *  @brief Interpolate forces from the fluid to the IBM nodes and spread forces from the IBM nodes back to the fluid.
 *  @param particlesNodes: Pointer to the IbmNodesSoA structure containing IBM node data.
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param step: The current simulation time step for collision checking.
 */
__global__
void ibmForceInterpolationSpread(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    dfloat *fMom,
    unsigned int step
);

/**
 *  @brief 
 *  @param particlesNodes: Pointer to the IbmNodesSoA structure containing IBM node data.
 *  @param pArray: Pointer to the array of ParticleCenter objects.
 *  @param firstIndex: The first index of the particle array to be processed.
 *  @param lastIndex: The last index of the particle array to be processed.
 *  @param step: The current simulation time step for collision checking.
 */
__global__
void ibmParticleNodeMovement(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif



