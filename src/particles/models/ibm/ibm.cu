#ifdef PARTICLE_MODEL

#include "ibm.cuh"

// Functions for the immersed boundary method
//void gpuForceInterpolationSpread();
//gpuResetNodesForces
//gpuParticleNodeMovement


void ibmSimulation(
    ParticlesSoA particles,
    cudaStream_t streamParticles,
    unsigned int step
){}


#endif //PARTICLE_MODEL