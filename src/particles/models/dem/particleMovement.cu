//#ifdef PARTICLE_MODEL

//functions related to the rigid body body of the particle and discretization

#include "particleMovement.cuh"

__global__ 
void gpuUpdateParticleCenterVelocityAndRotation(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);

    #ifdef IBM_DEBUG
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 f  x: %f y: %f z: %f\n",pc->f.x,pc->f.y,pc->f.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 f_old  x: %f y: %f z: %f\n",pc->f_old.x,pc->f_old.y,pc->f_old.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 dP_internal  x: %f y: %f z: %f\n",pc->dP_internal.x,pc->dP_internal.y,pc->dP_internal.z);
    #endif

    if(!pc->getMovable())
        return;

    const dfloat inv_volume = 1 / pc->getVolume();

    // Update particle center velocity using its surface forces and the body forces
    
    pc->setVelX(pc->getVelOldX() + (( (pc->getFOldX() * (1.0 - IBM_MOVEMENT_DISCRETIZATION ) 
                + pc->getFX() * IBM_MOVEMENT_DISCRETIZATION) + pc->getDPInternalX()) * inv_volume 
                + (pc->getDensity() - FLUID_DENSITY) * GX) / (pc->getDensity()));
    pc->setVelY(pc->getVelOldY() + (( (pc->getFOldY() * (1.0 - IBM_MOVEMENT_DISCRETIZATION )  
                + pc->getFY() * IBM_MOVEMENT_DISCRETIZATION) + pc->getDPInternalY()) * inv_volume 
                + (pc->getDensity() - FLUID_DENSITY) * GY) / (pc->getDensity()));
    pc->setVelZ(pc->getVelOldZ() + (( (pc->getFOldZ() * (1.0 - IBM_MOVEMENT_DISCRETIZATION )  
                + pc->getFZ() * IBM_MOVEMENT_DISCRETIZATION) + pc->getDPInternalZ()) * inv_volume 
                + (pc->getDensity() - FLUID_DENSITY) * GZ) / (pc->getDensity()));

    // Auxiliary variables for angular velocity update
    dfloat error = 1;
    dfloat3 wNew = dfloat3(), wAux;
    const dfloat3 M = pc->getM();
    const dfloat3 M_old = pc->getM_old();
    const dfloat3 w_old = pc->getW_old();
    dfloat6 I = pc->getI();

    dfloat6 Iaux6;
    dfloat4 q_rot;

    wAux.x = w_old.x;
    wAux.y = w_old.y;
    wAux.z = w_old.z;

    //dfloat I_det_neg = (I.zz*I.xy*I.xy + I.yy*I.xz*I.xz + I.xx*I.yz*I.yz - I.xx*I.yy*I.zz - 2*I.xy*I.xz*I.yz);
    dfloat inv_I_det_neg = 1.0/(I.zz*I.xy*I.xy + I.yy*I.xz*I.xz + I.xx*I.yz*I.yz - I.xx*I.yy*I.zz - 2*I.xy*I.xz*I.yz);
    dfloat3 wAvg, LM_avg, M_avg;

    wAvg.x = (w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION);
    wAvg.y = (w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION);
    wAvg.z = (w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION);

    LM_avg.x = pc->getDLInternalX() + (M.x * IBM_MOVEMENT_DISCRETIZATION + M_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    LM_avg.y = pc->getDLInternalY() + (M.y * IBM_MOVEMENT_DISCRETIZATION + M_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
    LM_avg.z = pc->getDLInternalZ() + (M.z * IBM_MOVEMENT_DISCRETIZATION + M_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION));

    //OLD CODE
    // Iteration process to upadate angular velocity 
    // (Crank-Nicolson implicit scheme)
    //for (int i = 0; error > 1e-4; i++)
    {
        //TODO the last term should be present in dL equation, but since it does not affect spheres, it will stay for now.
        /*
        wNew.x = pc->getWOldX() + (((M.x * IBM_MOVEMENT_DISCRETIZATION + M_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->getDLInternalX()) 
                - (I.zz - I.yy)*(w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION ) 
                               *(w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION))/I.xx;
        wNew.y = pc->getWOldY() + (((M.y * IBM_MOVEMENT_DISCRETIZATION + M_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->getDLInternalY()) 
                - (I.xx - I.zz)*(w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION ) 
                               *(w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION))/I.yy;
        wNew.z = pc->getWOldZ() + (((M.z * IBM_MOVEMENT_DISCRETIZATION + M_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION)) + pc->getDLInternalZ()) 
                - (I.yy - I.xx)*(w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION ) 
                               *(w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION))/I.zz;
        */


        wNew.x = pc->getWOldX() + ((I.yz*I.yz - I.yy*I.zz)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z))
                              - (I.xy*I.yz - I.xz*I.yy)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
                              - (I.xz*I.yz - I.xy*I.zz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z)))*inv_I_det_neg;
        wNew.y = pc->getWOldY() + ((I.xz*I.xz - I.xx*I.zz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z))
                              - (I.xy*I.xz - I.xx*I.yz)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
                              - (I.xz*I.yz - I.xy*I.zz)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z)))*inv_I_det_neg;
        wNew.z = pc->getWOldZ() + ((I.xy*I.xy - I.xx*I.yy)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
                              - (I.xy*I.xz - I.xx*I.yz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z))
                              - (I.xy*I.yz - I.xz*I.yy)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z)))*inv_I_det_neg;
                              
        //inertia update
        wAvg.x = (w_old.x * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.x * IBM_MOVEMENT_DISCRETIZATION);
        wAvg.y = (w_old.y * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.y * IBM_MOVEMENT_DISCRETIZATION);
        wAvg.z = (w_old.z * (1.0 - IBM_MOVEMENT_DISCRETIZATION) + wAux.z * IBM_MOVEMENT_DISCRETIZATION);

        //calculate rotation quartention
        q_rot = axis_angle_to_quart(wAvg,vector_length(wAvg));

        //compute new moment of inertia       
        Iaux6 = rotate_inertia_by_quart(q_rot,I);

        error =  (Iaux6.xx-I.xx)*(Iaux6.xx-I.xx)/(Iaux6.xx*Iaux6.xx);
        error += (Iaux6.yy-I.yy)*(Iaux6.yy-I.yy)/(Iaux6.yy*Iaux6.yy);
        error += (Iaux6.zz-I.zz)*(Iaux6.zz-I.zz)/(Iaux6.zz*Iaux6.zz);
        error += (Iaux6.xy-I.xy)*(Iaux6.xy-I.xy)/(Iaux6.xy*Iaux6.xy);
        error += (Iaux6.xz-I.xz)*(Iaux6.xz-I.xz)/(Iaux6.xz*Iaux6.xz);
        error += (Iaux6.yz-I.yz)*(Iaux6.yz-I.yz)/(Iaux6.yz*Iaux6.yz);

        //printf("error: %e \n",error);

        wAux.x = wNew.x;
        wAux.y = wNew.y;
        wAux.z = wNew.z;

        I.xx = Iaux6.xx;
        I.yy = Iaux6.yy;
        I.zz = Iaux6.zz;
        I.xy = Iaux6.xy;
        I.xz = Iaux6.xz;
        I.yz = Iaux6.yz;
    
       }

    // Store new velocities in particle center
    pc->setWX(wNew.x);
    pc->setWY(wNew.y);
    pc->setWZ(wNew.z);

    pc->setIXX(Iaux6.xx);
    pc->setIYY(Iaux6.yy);
    pc->setIZZ(Iaux6.zz);
    pc->setIXY(Iaux6.xy);
    pc->setIXZ(Iaux6.xz);
    pc->setIYZ(Iaux6.yz);

    #ifdef IBM_DEBUG
    printf("gpuUpdateParticleCenterVelocityAndRotation 2 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 2 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 2 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
    #endif
}

__global__
void gpuParticleMovement(
    ParticleCenter particleCenters[NUM_PARTICLES])
{
    unsigned int p = threadIdx.x + blockDim.x * blockIdx.x;

    if(p >= NUM_PARTICLES)
        return;

    ParticleCenter *pc = &(particleCenters[p]);

    #ifdef IBM_DEBUG
    printf("gpuParticleMovement 1 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
    printf("gpuParticleMovement 1 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
    printf("gpuParticleMovement 1 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
    #endif
    
    
    if(!pc->getMovable())
        return;

    #ifdef IBM_BC_X_WALL
        pc->setPosX((pc->getPosX() + (pc->getVelX() * IBM_MOVEMENT_DISCRETIZATION + pc->getVelOldX() * (1.0 - IBM_MOVEMENT_DISCRETIZATION))));
    #endif //IBM_BC_X_WALL
    #ifdef IBM_BC_X_PERIODIC
        dfloat dx =  (pc->getVelX() * IBM_MOVEMENT_DISCRETIZATION + pc->getVelOldX() * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->setPosX(IBM_BC_X_0 + std::fmod((dfloat)(pc->getPosX() + dx + IBM_BC_X_E - IBM_BC_X_0 - IBM_BC_X_0) , (dfloat)(IBM_BC_X_E - IBM_BC_X_0))); 
    #endif //IBM_BC_X_PERIODIC

    #ifdef IBM_BC_Y_WALL
        pc->setPosY((pc->getPosY() + (pc->getVelY() * IBM_MOVEMENT_DISCRETIZATION + pc->getVelOldY() * (1.0 - IBM_MOVEMENT_DISCRETIZATION))));
    #endif //IBM_BC_Y_WALL
    #ifdef IBM_BC_Y_PERIODIC
        dfloat dy =  (pc->getVelY() * IBM_MOVEMENT_DISCRETIZATION + pc->getVelOldY() * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->setPosY(IBM_BC_Y_0 + std::fmod((dfloat)(pc->getPosY() + dy + IBM_BC_Y_E - IBM_BC_Y_0 - IBM_BC_Y_0) , (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0)));
    #endif //IBM_BC_Y_PERIODIC

    // #ifdef IBM_BC_Z_WALL
        pc->setPosZ((pc->getPosZ() + (pc->getVelZ() * IBM_MOVEMENT_DISCRETIZATION + pc->getVelOldZ() * (1.0 - IBM_MOVEMENT_DISCRETIZATION))));
    // #endif //IBM_BC_Z_WALL
    #ifdef IBM_BC_Z_PERIODIC
        dfloat dz =  (pc->getVelZ() * IBM_MOVEMENT_DISCRETIZATION + pc->getVelOldZ() * (1.0 - IBM_MOVEMENT_DISCRETIZATION));
        pc->setPosZ(IBM_BC_Z_0 + std::fmod((dfloat)(pc->getPosZ() + dz + IBM_BC_Z_E - IBM_BC_Z_0 - IBM_BC_Z_0) , (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0))); 
    #endif //IBM_BC_Z_PERIODIC

    pc->setWAvgX((pc->getWX()   * IBM_MOVEMENT_DISCRETIZATION + pc->getWOldX()   * (1.0 - IBM_MOVEMENT_DISCRETIZATION)));
    pc->setWAvgY((pc->getWY()   * IBM_MOVEMENT_DISCRETIZATION + pc->getWOldY()   * (1.0 - IBM_MOVEMENT_DISCRETIZATION)));
    pc->setWAvgZ((pc->getWZ()   * IBM_MOVEMENT_DISCRETIZATION + pc->getWOldZ()   * (1.0 - IBM_MOVEMENT_DISCRETIZATION)));
    pc->setWPosX(pc->getWAvgX());
    pc->setWPosY(pc->getWAvgY());
    pc->setWPosZ(pc->getWAvgZ());

    #ifdef IBM_DEBUG
    printf("gpuParticleMovement 2 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
    printf("gpuParticleMovement 2 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
    printf("gpuParticleMovement 2 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
    #endif


    //update orientation vector
    const dfloat w_norm = sqrt((pc->getWAvgX() * pc->getWAvgX()) 
                             + (pc->getWAvgY() * pc->getWAvgY()) 
                             + (pc->getWAvgZ() * pc->getWAvgZ()));
    const dfloat q0 = cos(0.5*w_norm);
    const dfloat qi = (pc->getWAvgX()/w_norm) * sin (0.5*w_norm);
    const dfloat qj = (pc->getWAvgY()/w_norm) * sin (0.5*w_norm);
    const dfloat qk = (pc->getWAvgZ()/w_norm) * sin (0.5*w_norm);
    const dfloat tq0m1 = (q0*q0) - 0.5;

    dfloat x_vec = pc->getSemiAxis1X() - pc->getPosOldX();
    dfloat y_vec = pc->getSemiAxis1Y() - pc->getPosOldY();
    dfloat z_vec = pc->getSemiAxis1Z() - pc->getPosOldZ();

    
    pc->setSemiAxis1X(pc->getPosX() + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec));
    pc->setSemiAxis1Y(pc->getPosY() + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec));
    pc->setSemiAxis1Z(pc->getPosZ() + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec));

    x_vec = pc->getSemiAxis2X() - pc->getPosOldX();
    y_vec = pc->getSemiAxis2Y() - pc->getPosOldY();
    z_vec = pc->getSemiAxis2Z() - pc->getPosOldZ();

    pc->setSemiAxis2X(pc->getPosX() +  2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec));
    pc->setSemiAxis2Y(pc->getPosY() +  2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1  + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec));
    pc->setSemiAxis2Z(pc->getPosZ() +  2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1  + (qk*qk))*z_vec));

    x_vec = pc->getSemiAxis3X() - pc->getPosOldX();
    y_vec = pc->getSemiAxis3Y() - pc->getPosOldY();
    z_vec = pc->getSemiAxis3Z() - pc->getPosOldZ();

    pc->setSemiAxis3X(pc->getPosX() +  2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec));
    pc->setSemiAxis3X(pc->getPosY() +  2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1  + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec));
    pc->setSemiAxis3X(pc->getPosZ() +  2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1  + (qk*qk))*z_vec));

}


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
    pc->setDPInternalX(0.0); //RHO_0 * pc->getVolume() * (pc->getVelX() - pc->getVelOldX());
    pc->setDPInternalY(0.0); //RHO_0 * pc->getVolume() * (pc->getVelY() - pc->getVelOldY());
    pc->setDPInternalY(0.0); //RHO_0 * pc->getVolume() * (pc->getVelZ() - pc->getVelOldZ());

    // Internal angular momentum delta = (rho_f/rho_p)*I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    
    pc->setDLInternalX(0.0); //(RHO_0 / pc->getDensity()) * pc->getIXX() * (pc->getWX() - pc->getWOldX());
    pc->setDLInternalX(0.0); //(RHO_0 / pc->getDensity()) * pc->getIYY() * (pc->getWY() - pc->getWOldY());
    pc->setDLInternalX(0.0); //(RHO_0 / pc->getDensity()) * pc->getIZZ() * (pc->getWZ() - pc->getWOldZ());

    pc->setPosOldX(pc->getPosX());
    pc->setPosOldY(pc->getPosY());
    pc->setPosOldZ(pc->getPosZ());

    pc->setVelOldX(pc->getVelX());
    pc->setVelOldY(pc->getVelY());
    pc->setVelOldZ(pc->getVelZ());

    pc->setWOldX(pc->getWX());
    pc->setWOldY(pc->getWY());
    pc->setWOldZ(pc->getWZ());

    pc->setFOldX(pc->getFX());
    pc->setFOldY(pc->getFY());
    pc->setFOldZ(pc->getFZ());

    // Reset force, because kernel is always added
    pc->setFX(0);
    pc->setFY(0);
    pc->setFZ(0);

    pc->setMX(0);
    pc->setMY(0);
    pc->setMZ(0);
}



//#endif //PARTICLE_MODEL