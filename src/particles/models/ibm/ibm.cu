// #ifdef PARTICLE_MODEL

#include "ibm.cuh"
// Functions for the immersed boundary method
//void gpuForceInterpolationSpread();
//gpuResetNodesForces
//gpuParticleNodeMovement


__global__ 
void gpuResetNodesForces(IbmNodesSoA* particlesNodes)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= particlesNodes->getNumNodes())
        return;

    const dfloat3SoA force = particlesNodes->getF();
    const dfloat3SoA delta_force = particlesNodes->getDeltaF();

    force.x[idx] = 0;
    force.y[idx] = 0;
    force.z[idx] = 0;
    delta_force.x[idx] = 0;
    delta_force.y[idx] = 0;
    delta_force.z[idx] = 0;
}


void ibmSimulation(
    ParticlesSoA particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
){
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    gpuUpdateParticleOldValues<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(
        particles.getPCenterArray());
    checkCudaErrors(cudaStreamSynchronize(streamParticles));

     // Grid for only  z-borders
    //  dim3 copyMacrGrid = gridLBM;
     // Grid for full domain, including z-borders
    //  dim3 borderMacrGrid = gridLBM; 
     // Only 1 in z
    //  copyMacrGrid.z = MACR_BORDER_NODES;
    //  borderMacrGrid.z += MACR_BORDER_NODES*2;
 
    unsigned int gridNodesIBM[N_GPUS];
    unsigned int threadsNodesIBM[N_GPUS];
    for(int i = 0; i < N_GPUS; i++){
        threadsNodesIBM[i] = 64;
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        unsigned int pNumNodes = particles.getNodesSoA()->getNumNodes();
        gridNodesIBM[i] = pNumNodes % threadsNodesIBM[i] ? pNumNodes / threadsNodesIBM[i] + 1 : pNumNodes / threadsNodesIBM[i];
    }
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));

    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        //int nxt = (i+1) % N_GPUS;
        //Copy macroscopics
        //gpuCopyBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[i]>>>(macr[i], macr[nxt]); Verificar se é necessário
        checkCudaErrors(cudaStreamSynchronize(streamParticles));
        getLastCudaError("Copy macroscopics border error\n");
        // If GPU has nodes in it
        if(particles.getNodesSoA()->getNumNodes() > 0){
            // Reset forces in all IBM nodes;
            gpuResetNodesForces<<<gridNodesIBM[i], threadsNodesIBM[i], 0, streamParticles>>>(particles.getNodesSoA());
            checkCudaErrors(cudaStreamSynchronize(streamParticles));
            getLastCudaError("Reset IBM nodes forces error\n");
        }
    }

     // Calculate collision force between particles
     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
     checkCudaErrors(cudaStreamSynchronize(streamParticles)); 
 
     // First update particle velocity using body center force and constant forces
     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
     gpuUpdateParticleCenterVelocityAndRotation <<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles >>>(
         particles.getPCenterArray());
     getLastCudaError("IBM update particle center velocity error\n");
     checkCudaErrors(cudaStreamSynchronize(streamParticles));
 
    //  for (int i = 0; i < IBM_MAX_ITERATION; i++)
    //  {
    //      for(int j = 0; j < N_GPUS; j++){
    //          // If GPU has nodes in it
    //          if(particles.nodesSoA[j].numNodes > 0){
    //              checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
    //              // Make the interpolation of LBM and spreading of IBM forces
    //              gpuForceInterpolationSpread<<<gridNodesIBM[j], threadsNodesIBM[j], 
    //                  0, streamIBM[j]>>>(
    //                  particles.nodesSoA[j], particles.pCenterArray, macr[j], ibmMacrsAux, j);
    //              checkCudaErrors(cudaStreamSynchronize(streamIBM[j]));
    //              getLastCudaError("IBM interpolation spread error\n");
    //          }
    //      }
 
    //      checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    //      // Update particle velocity using body center force and constant forces
    //      // Migrar
    //      gpuUpdateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0]>>>(
    //          particles.pCenterArray);
    //      checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
    //      getLastCudaError("IBM update particle center velocity error\n");
 
    //      // Sum border macroscopics
    //      // for(int j = 0; j < N_GPUS; j++){
    //      //     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
    //      //     int nxt = (j+1) % N_GPUS;
    //      //     int prv = (j-1+N_GPUS) % N_GPUS;
    //      //     bool run_nxt = nxt != 0;
    //      //     bool run_prv = prv != (N_GPUS-1);
    //      //     #ifdef IBM_BC_Z_PERIODIC
    //      //     run_nxt = true;
    //      //     run_prv = true;
    //      //     #endif
             
    //      //     if(run_nxt){
    //      //         gpuSumBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[nxt], ibmMacrsAux, j, 1);
    //      //         checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
    //      //     }
    //      //     if(run_prv){
    //      //         gpuSumBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[prv], ibmMacrsAux, j, -1);
    //      //         checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
    //      //     }
    //      //     getLastCudaError("Sum border macroscopics error\n");
    //      // }
 
    //      // #if IBM_EULER_OPTIMIZATION
 
    //      // for(int j = 0; j < N_GPUS; j++){
    //      //     if(pEulerNodes->currEulerNodes[j] > 0){
    //      //         checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
    //      //         dim3 currGrid(pEulerNodes->currEulerNodes[j]/64+(pEulerNodes->currEulerNodes[j]%64? 1 : 0), 1, 1);
    //      //         gpuEulerSumIBMAuxsReset<<<currGrid, 64, 0, streamLBM[j]>>>(macr[j], ibmMacrsAux,
    //      //             pEulerNodes->eulerIndexesUpdate[j], pEulerNodes->currEulerNodes[j], j);
    //      //         checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
    //      //         getLastCudaError("IBM sum auxiliary values error\n");
    //      //     }
    //      // }
    //      // #else
    //      // for(int j = 0; j < N_GPUS; j++){
    //      //     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
    //      //     gpuEulerSumIBMAuxsReset<<<borderMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[j], ibmMacrsAux, j);
    //      //     checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
    //      // }
    //      // #endif
 
    //  }

    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    // Update particle center position and its old values
    // Migrar
    gpuParticleMovement<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(
        particles.getPCenterArray());
    checkCudaErrors(cudaStreamSynchronize(streamParticles));
    getLastCudaError("IBM particle movement error\n");

    // for(int i = 0; i < N_GPUS; i++){
    //     // If GPU has nodes in it
    //     if(particles.getNodesSoA()[i].getNumNodes() > 0){ // particles.nodesSoA[i].numNodes
    //         checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
    //         // Update particle nodes positions
    //         gpuParticleNodeMovement<<<gridNodesIBM[i], threadsNodesIBM[i], 0, streamParticles>>>(
    //             particles.getNodesSoA()[i], particles.getPCenterArray());
    //         checkCudaErrors(cudaStreamSynchronize(streamParticles));
    //         getLastCudaError("IBM particle movement error\n");
    //     }
    // }

    checkCudaErrors(cudaDeviceSynchronize());
 
}

// #endif //PARTICLE_MODEL