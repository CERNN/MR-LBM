

#include "particleSim.cuh"

#ifdef PARTICLE_MODEL

void particleSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t *streamParticles,
    unsigned int step
){

    int numIBM    = particles->getMethodCount(IBM);
    int numPIBM   = particles->getMethodCount(PIBM);
    int numTRACER = particles->getMethodCount(TRACER);

    if(numIBM>0){
       ibmSimulation(particles,fMom,streamParticles[0],step);
    }
    if(numPIBM>0){
        pibmSimulation(particles,fMom,streamParticles[0],step);
    }
    if(numTRACER>0){
        tracerSimulation(particles,fMom,streamParticles[0],step);
    }

}

#endif //PARTICLE_MODEL