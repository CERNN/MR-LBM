#ifndef __PARTICLE_SIM_CUH
#define __PARTICLE_SIM_CUH

#include "./../class/particle.cuh"

//models
#include "ibm/ibm.cuh"
#include "pibm/pibm.cuh"
#include "tracer/tracer.cuh"

// void ibmSimulation(
//     ParticlesSoA *particles,
//     dfloat *fMom,
//     cudaStream_t *streamParticles,
//     unsigned int step
// );

void particleSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t *streamParticles,
    unsigned int step
);

#endif //__PARTICLE_SIM_CUH