#ifndef __PARTICLE_SIM_CUH
#define __PARTICLE_SIM_CUH

#include "./../class/Particle.cuh"

//models
#include "ibm/ibm.cuh"
#include "pibm/pibm.cuh"
#include "tracer/tracer.cuh"

void particleSimulation(
        ParticlesSoA particles
);

#endif //__PARTICLE_SIM_CUH