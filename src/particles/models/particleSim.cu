#include "particleSim.cuh"

void particleSimulation(
        ParticlesSoA particles
){

    int numIBM    = particles.getMethodCount(IBM);
    int numPIBM   = particles.getMethodCount(PIBM);
    int numTRACER = particles.getMethodCount(TRACER);

    if(numIBM>0){
        /*code*/
    }
    if(numPIBM>0){
        /*code*/
    }
    if(numTRACER>0){
        /*code*/
    }


}
