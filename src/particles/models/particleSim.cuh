/**
 *  @file particleSim.cuh
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Handle particle simulations
 *  @version 0.1.0
 *  @date 01/09/2025
 */

#ifndef __PARTICLE_SIM_CUH
#define __PARTICLE_SIM_CUH

#include "./../class/particle.cuh"

//models
#include "ibm/ibm.cuh"
#include "pibm/pibm.cuh"
#include "tracer/tracer.cuh"
#include "dem/collision/collisionDetection.cuh"

#ifdef PARTICLE_MODEL

/**
 *  @brief Handle all simulation process of the particles
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param fMom: Pointer to the device array containing the current macroscopic moments.
 *  @param streamParticles: cuda stream for particles
 *  @param step: The current simulation time step for collision checking.
 */
void particleSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t *streamParticles,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif //__PARTICLE_SIM_CUH

