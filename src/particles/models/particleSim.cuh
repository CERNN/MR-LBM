/**
*   @file particleSim.cuh
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @author Ricardo de Souza
*   @brief Handle particle simulations
*   @version 0.1.0
*   @date 01/09/2025
*/

#ifndef __PARTICLE_SIM_CUH
#define __PARTICLE_SIM_CUH

#include "./../class/particle.cuh"

//models
#include "ibm/ibm.cuh"
#include "pibm/pibm.cuh"
#include "tracer/tracer.cuh"

#ifdef PARTICLE_MODEL
void particleSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t *streamParticles,
    unsigned int step
);

#endif //PARTICLE_MODEL
#endif //__PARTICLE_SIM_CUH

