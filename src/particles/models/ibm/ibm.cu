// #ifdef PARTICLE_MODEL

#include "ibm.cuh"

// Functions for the immersed boundary method
//void gpuForceInterpolationSpread();
//gpuResetNodesForces
//gpuParticleNodeMovement

__global__
void gpuUpdateParticleOldValues(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);

    // Internal linear momentum delta = rho*volume*delta(v)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    // pc->dP_internal.x = 0.0; //RHO_0 * pc->volume * (pc->vel.x - pc->vel_old.x);
    // pc->dP_internal.y = 0.0; //RHO_0 * pc->volume * (pc->vel.y - pc->vel_old.y);
    // pc->dP_internal.z = 0.0; //RHO_0 * pc->volume * (pc->vel.z - pc->vel_old.z);

    pc->setDPInternalX(0.0); //RHO_0 * pc->volume * (pc->vel.x - pc->vel_old.x);
    pc->setDPInternalY(0.0); //RHO_0 * pc->volume * (pc->vel.y - pc->vel_old.y);
    pc->setDPInternalY(0.0); //RHO_0 * pc->volume * (pc->vel.z - pc->vel_old.z);

    // Internal angular momentum delta = (rho_f/rho_p)*I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    // pc->dL_internal.x = 0.0; //(RHO_0 / pc->density) * pc->I.xx * (pc->w.x - pc->w_old.x);
    // pc->dL_internal.y = 0.0; //(RHO_0 / pc->density) * pc->I.yy * (pc->w.y - pc->w_old.y);
    // pc->dL_internal.z = 0.0; //(RHO_0 / pc->density) * pc->I.zz * (pc->w.z - pc->w_old.z);

    pc->setDLInternalX(0.0);
    pc->setDLInternalX(0.0);
    pc->setDLInternalX(0.0);

    // pc->pos_old.x = pc->pos.x;
    // pc->pos_old.y = pc->pos.y;
    // pc->pos_old.z = pc->pos.z;
    pc->setPosOldX(pc->getPosX());
    pc->setPosOldY(pc->getPosY());
    pc->setPosOldZ(pc->getPosZ());

    // pc->vel_old.x = pc->vel.x;
    // pc->vel_old.y = pc->vel.y;
    // pc->vel_old.z = pc->vel.z;
    pc->setVelOldX(pc->getVelX());
    pc->setVelOldY(pc->getVelY());
    pc->setVelOldZ(pc->getVelZ());

    // pc->w_old.x = pc->w.x;
    // pc->w_old.y = pc->w.y;
    // pc->w_old.z = pc->w.z;
    pc->setWOldX(pc->getWX());
    pc->setWOldY(pc->getWY());
    pc->setWOldZ(pc->getWZ());

    // pc->f_old.x = pc->f.x;
    // pc->f_old.y = pc->f.y;
    // pc->f_old.z = pc->f.z;
    pc->setFOldX(pc->getFX());
    pc->setFOldY(pc->getFY());
    pc->setFOldZ(pc->getFZ());

    // Reset force, because kernel is always added
    // pc->f.x = 0;
    // pc->f.y = 0;
    // pc->f.z = 0;
    pc->setFX(0);
    pc->setFY(0);
    pc->setFZ(0);

    // pc->M.x = 0;
    // pc->M.y = 0;
    // pc->M.z = 0;
    pc->setMX(0);
    pc->setMY(0);
    pc->setMZ(0);
}

// __global__ 
// void gpuResetNodesForces(ParticleNodeSoA* particlesNodes)
// {
//     int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     if (idx >= particlesNodes->getNumNodes())
//         return;

//     const dfloat3SoA force = particlesNodes->getF();
//     const dfloat3SoA delta_force = particlesNodes->getDeltaF();

//     force.x[idx] = 0;
//     force.y[idx] = 0;
//     force.z[idx] = 0;
//     delta_force.x[idx] = 0;
//     delta_force.y[idx] = 0;
//     delta_force.z[idx] = 0;
// }

// __global__ 
// void gpuUpdateParticleCenterVelocityAndRotation(
//     ParticleCenter particleCenters[NUM_PARTICLES])
// {
//     unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

//     if(p >= NUM_PARTICLES)
//         return;

//     ParticleCenter *pc = &(particleCenters[p]);

//     #ifdef IBM_DEBUG
//     printf("gpuUpdateParticleCenterVelocityAndRotation 1 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 1 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 1 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 1 f  x: %f y: %f z: %f\n",pc->f.x,pc->f.y,pc->f.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 1 f_old  x: %f y: %f z: %f\n",pc->f_old.x,pc->f_old.y,pc->f_old.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 1 dP_internal  x: %f y: %f z: %f\n",pc->dP_internal.x,pc->dP_internal.y,pc->dP_internal.z);
//     #endif

//     if(!pc->getMovable())
//         return;

//     const dfloat inv_volume = 1 / pc->getVolume();

//     // Update particle center velocity using its surface forces and the body forces
//     pc->vel.x = pc->vel_old.x + (( (pc->f_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION ) 
//                 + pc->f.x * IBM_MOVEMENT_DISCRETIZATION) + pc->dP_internal.x) * inv_volume 
//                 + (pc->density - FLUID_DENSITY) * GX) / (pc->density);
//     pc->vel.y = pc->vel_old.y + (( (pc->f_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION )  
//                 + pc->f.y * IBM_MOVEMENT_DISCRETIZATION) + pc->dP_internal.y) * inv_volume 
//                 + (pc->density - FLUID_DENSITY) * GY) / (pc->density);
//     pc->vel.z = pc->vel_old.z + (( (pc->f_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION )  
//                 + pc->f.z * IBM_MOVEMENT_DISCRETIZATION) + pc->dP_internal.z) * inv_volume 
//                 + (pc->density - FLUID_DENSITY) * GZ) / (pc->density);

//     // Auxiliary variables for angular velocity update
//     dfloat error = 1;
//     dfloat3 wNew = dfloat3(), wAux;
//     const dfloat3 M = pc->M;
//     const dfloat3 M_old = pc->M_old;
//     const dfloat3 w_old = pc->w_old;
//     dfloat6 I = pc->I;

//     dfloat6 Iaux6;
//     dfloat4 q_rot;

//     wAux.x = w_old.x;
//     wAux.y = w_old.y;
//     wAux.z = w_old.z;

//     dfloat I_det_neg = (I.zz*I.xy*I.xy + I.yy*I.xz*I.xz + I.xx*I.yz*I.yz - I.xx*I.yy*I.zz - 2*I.xy*I.xz*I.yz);
//     dfloat inv_I_det_neg = 1.0/(I.zz*I.xy*I.xy + I.yy*I.xz*I.xz + I.xx*I.yz*I.yz - I.xx*I.yy*I.zz - 2*I.xy*I.xz*I.yz);
//     dfloat3 wAvg, LM_avg, M_avg;

//     wAvg.x = (w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION);
//     wAvg.y = (w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION);
//     wAvg.z = (w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION);

//     LM_avg.x = pc->dL_internal.x + (M.x * IBM_MOVEMENT_DISCRETIZATION + M_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
//     LM_avg.y = pc->dL_internal.y + (M.y * IBM_MOVEMENT_DISCRETIZATION + M_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
//     LM_avg.z = pc->dL_internal.z + (M.z * IBM_MOVEMENT_DISCRETIZATION + M_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION));

//     //OLD CODE
//     // Iteration process to upadate angular velocity 
//     // (Crank-Nicolson implicit scheme)
//     //for (int i = 0; error > 1e-4; i++)
//     {
//         //TODO the last term should be present in dL equation, but since it does not affect spheres, it will stay for now.
//         /*
//         wNew.x = pc->w_old.x + (((M.x * IBM_MOVEMENT_DISCRETIZATION + M_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->dL_internal.x) 
//                 - (I.zz - I.yy)*(w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION ) 
//                                *(w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION))/I.xx;
//         wNew.y = pc->w_old.y + (((M.y * IBM_MOVEMENT_DISCRETIZATION + M_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->dL_internal.y) 
//                 - (I.xx - I.zz)*(w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION ) 
//                                *(w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION))/I.yy;
//         wNew.z = pc->w_old.z + (((M.z * IBM_MOVEMENT_DISCRETIZATION + M_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->dL_internal.z) 
//                 - (I.yy - I.xx)*(w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION ) 
//                                *(w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION))/I.zz;
//         */


//         wNew.x = pc->w_old.x + ((I.yz*I.yz - I.yy*I.zz)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z))
//                               - (I.xy*I.yz - I.xz*I.yy)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
//                               - (I.xz*I.yz - I.xy*I.zz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z)))*inv_I_det_neg;
//         wNew.y = pc->w_old.y + ((I.xz*I.xz - I.xx*I.zz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z))
//                               - (I.xy*I.xz - I.xx*I.yz)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
//                               - (I.xz*I.yz - I.xy*I.zz)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z)))*inv_I_det_neg;
//         wNew.z = pc->w_old.z + ((I.xy*I.xy - I.xx*I.yy)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
//                               - (I.xy*I.xz - I.xx*I.yz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z))
//                               - (I.xy*I.yz - I.xz*I.yy)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z)))*inv_I_det_neg;
                              
//         //inertia update
//         wAvg.x = (w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION);
//         wAvg.y = (w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION);
//         wAvg.z = (w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION);

//         //calculate rotation quartention
//         q_rot = axis_angle_to_quart(wAvg,vector_length(wAvg));

//         //compute new moment of inertia       
//         Iaux6 = rotate_inertia_by_quart(q_rot,I);

//         error =  (Iaux6.xx-I.xx)*(Iaux6.xx-I.xx)/(Iaux6.xx*Iaux6.xx);
//         error += (Iaux6.yy-I.yy)*(Iaux6.yy-I.yy)/(Iaux6.yy*Iaux6.yy);
//         error += (Iaux6.zz-I.zz)*(Iaux6.zz-I.zz)/(Iaux6.zz*Iaux6.zz);
//         error += (Iaux6.xy-I.xy)*(Iaux6.xy-I.xy)/(Iaux6.xy*Iaux6.xy);
//         error += (Iaux6.xz-I.xz)*(Iaux6.xz-I.xz)/(Iaux6.xz*Iaux6.xz);
//         error += (Iaux6.yz-I.yz)*(Iaux6.yz-I.yz)/(Iaux6.yz*Iaux6.yz);

//         //printf("error: %e \n",error);

//         wAux.x = wNew.x;
//         wAux.y = wNew.y;
//         wAux.z = wNew.z;

//         I.xx = Iaux6.xx;
//         I.yy = Iaux6.yy;
//         I.zz = Iaux6.zz;
//         I.xy = Iaux6.xy;
//         I.xz = Iaux6.xz;
//         I.yz = Iaux6.yz;
    
//        }

//     // Store new velocities in particle center
//     pc->w.x = wNew.x;
//     pc->w.y = wNew.y;
//     pc->w.z = wNew.z;

//     pc->I.xx = Iaux6.xx;
//     pc->I.yy = Iaux6.yy;
//     pc->I.zz = Iaux6.zz;
//     pc->I.xy = Iaux6.xy;
//     pc->I.xz = Iaux6.xz;
//     pc->I.yz = Iaux6.yz;

//     #ifdef IBM_DEBUG
//     printf("gpuUpdateParticleCenterVelocityAndRotation 2 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 2 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
//     printf("gpuUpdateParticleCenterVelocityAndRotation 2 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
//     #endif
// }


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

    // for(int i = 0; i < N_GPUS; i++){
    //     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
    //     int nxt = (i+1) % N_GPUS;
    //     // Copy macroscopics
    //    // gpuCopyBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[i]>>>(macr[i], macr[nxt]); Verificar se é necessário
    //     checkCudaErrors(cudaStreamSynchronize(streamParticles[i]));
    //     getLastCudaError("Copy macroscopics border error\n");
    //     // If GPU has nodes in it
    //     if(particles.getNodesSoA()->getNumNodes() > 0){
    //         // Reset forces in all IBM nodes;
    //         gpuResetNodesForces<<<gridNodesIBM[i], threadsNodesIBM[i], 0, streamParticles[i]>>>(particles.getNodesSoA());
    //         checkCudaErrors(cudaStreamSynchronize(streamParticles[i]));
    //         getLastCudaError("Reset IBM nodes forces error\n");
    //     }
    // }

    //  // Calculate collision force between particles
    //  checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    //  checkCudaErrors(cudaStreamSynchronize(streamParticles[0])); 
 
    //  // First update particle velocity using body center force and constant forces
    //  checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    //  gpuUpdateParticleCenterVelocityAndRotation <<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamIBM[0] >>>(
    //      particles.pCenterArray);
    //  getLastCudaError("IBM update particle center velocity error\n");
    //  checkCudaErrors(cudaStreamSynchronize(streamIBM[0]));
 
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
 
}

// #endif //PARTICLE_MODEL