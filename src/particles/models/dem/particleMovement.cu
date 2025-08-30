//#ifdef PARTICLE_MODEL

//functions related to the rigid body body of the particle and discretization

#include "particleMovement.cuh"

__global__
void gpuUpdateParticleOldValues(
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex)
{
    unsigned int localIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globalIdx = firstIndex + localIdx;

    if (globalIdx < firstIndex || globalIdx > lastIndex || globalIdx >= NUM_PARTICLES) {
        return;
    }

    if (pArray == nullptr) {
        printf("ERROR: particles is nullptr\n");
        return;
    }

    if (globalIdx < firstIndex || globalIdx > lastIndex || globalIdx >= NUM_PARTICLES) {
        return;
    }

    ParticleCenter* pc_i = &pArray[globalIdx];

    // Internal linear momentum delta = rho*volume*delta(v)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    pc_i->setDPInternalX(0.0); //RHO_0 * pc_i->getVolume() * (pc_i->getVelX() - pc_i->getVelOldX());
    pc_i->setDPInternalY(0.0); //RHO_0 * pc_i->getVolume() * (pc_i->getVelY() - pc_i->getVelOldY());
    pc_i->setDPInternalZ(0.0); //RHO_0 * pc_i->getVolume() * (pc_i->getVelZ() - pc_i->getVelOldZ());

    // Internal angular momentum delta = (rho_f/rho_p)*I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    
    pc_i->setDLInternalX(0.0); //(RHO_0 / pc_i->getDensity()) * pc_i->getIXX() * (pc_i->getWX() - pc_i->getWOldX());
    pc_i->setDLInternalY(0.0); //(RHO_0 / pc_i->getDensity()) * pc_i->getIYY() * (pc_i->getWY() - pc_i->getWOldY());
    pc_i->setDLInternalZ(0.0); //(RHO_0 / pc_i->getDensity()) * pc_i->getIZZ() * (pc_i->getWZ() - pc_i->getWOldZ());

    //printf("gpuUpdateParticleOldValues 2 pos  x: %f y: %f z: %f\n",pc_i->getPosOldX(),pc_i->getPosOldY(),pc_i->getPosOldZ());
    //printf("gpuUpdateParticleOldValues 3 pos  x: %f y: %f z: %f\n",pc_i->getVelOldX(),pc_i->getVelOldY(),pc_i->getVelOldZ());
    //printf("gpuUpdateParticleOldValues 4 pos  x: %f y: %f z: %f\n",pc_i->getWOldX(),pc_i->getWOldY(),pc_i->getWOldZ());
    //printf("gpuUpdateParticleOldValues 5 pos  x: %f y: %f z: %f\n",pc_i->getFOldX(),pc_i->getFOldY(),pc_i->getFOldZ());
    //printf("gpuUpdateParticleOldValues 6 pos  x: %f y: %f z: %f\n",pc_i->getFX(),pc_i->getFY(),pc_i->getFZ());
    //printf("gpuUpdateParticleOldValues 7 pos  x: %f y: %f z: %f\n",pc_i->getMX(),pc_i->getMY(),pc_i->getMZ());
    
    pc_i->setPos_old(pc_i->getPos());
    pc_i->setVel_old(pc_i->getVel());
    pc_i->setW_old(pc_i->getW());
    pc_i->setF_old(pc_i->getF());
    pc_i->setM_old(pc_i->getM());
    pc_i->setF(dfloat3(0,0,0));
    pc_i->setM(dfloat3(0,0,0));

}

__global__ 
void gpuUpdateParticleCenterVelocityAndRotation(
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex)
{
    unsigned int localIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int globalIdx = firstIndex + localIdx;

    if (globalIdx < firstIndex || globalIdx > lastIndex || globalIdx >= NUM_PARTICLES) {
        return;
    }

    if (pArray == nullptr) {
        printf("ERROR: particles is nullptr\n");
        return;
    }

    if (globalIdx < firstIndex || globalIdx > lastIndex || globalIdx >= NUM_PARTICLES) {
        return;
    }

    ParticleCenter* pc_i = &pArray[globalIdx];

    if(!pc_i->getMovable())
        return;

    #ifdef IBM_DEBUG
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 pos  x: %f y: %f z: %f\n",pc->pos.x,pc->pos.y,pc->pos.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 vel  x: %f y: %f z: %f\n",pc->vel.x,pc->vel.y,pc->vel.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 w  x: %f y: %f z: %f\n",pc->w.x,pc->w.y,pc->w.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 f  x: %f y: %f z: %f\n",pc->f.x,pc->f.y,pc->f.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 f_old  x: %f y: %f z: %f\n",pc->f_old.x,pc->f_old.y,pc->f_old.z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 dP_internal  x: %f y: %f z: %f\n",pc->dP_internal.x,pc->dP_internal.y,pc->dP_internal.z);
    #endif

    // Update particle center velocity using its surface forces and the body forces
    pc_i->setVel(pc_i->getVel_old() + ((0.5 * (pc_i->getF_old() + pc_i->getF()) + pc_i->getDP_internal())) / (pc_i->getVolume()) 
                + (1.0 - FLUID_DENSITY/pc_i->getDensity()) * dfloat3(GX,GY,GZ));
     
    // Update particle angular velocity  

    dfloat6 I = pc_i->getI();
    dfloat inv_I_det_neg = 1.0/(I.zz*I.xy*I.xy + I.yy*I.xz*I.xz + I.xx*I.yz*I.yz - I.xx*I.yy*I.zz - 2*I.xy*I.xz*I.yz);
    dfloat3 wAux = pc_i->getW_old();
    dfloat3 wAvg = 0.5*(pc_i->getW_old() + pc_i->getW());
    dfloat3 LM_avg = pc_i->getDL_internal() + 0.5*(pc_i->getM_old() + pc_i->getM());

    dfloat error = 1.0;
    dfloat3 wNew;
    dfloat4 q_rot;
    dfloat6 Iaux6;

    //for (int i = 0; error > 1e-4; i++)
    {
        wNew.x = pc_i->getWOldX() + ((I.yz*I.yz - I.yy*I.zz)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z))
                                - (I.xy*I.yz - I.xz*I.yy)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
                                - (I.xz*I.yz - I.xy*I.zz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z)))*inv_I_det_neg;
        wNew.y = pc_i->getWOldY() + ((I.xz*I.xz - I.xx*I.zz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z))
                                - (I.xy*I.xz - I.xx*I.yz)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
                                - (I.xz*I.yz - I.xy*I.zz)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z)))*inv_I_det_neg;
        wNew.z = pc_i->getWOldZ() + ((I.xy*I.xy - I.xx*I.yy)*(LM_avg.z + (wAvg.y)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z) - (wAvg.x)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z))
                                - (I.xy*I.xz - I.xx*I.yz)*(LM_avg.y + (wAvg.x)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z) - (wAvg.z)*(I.xx*wAvg.x + I.xy*wAvg.y + I.xz*wAvg.z))
                                - (I.xy*I.yz - I.xz*I.yy)*(LM_avg.x + (wAvg.z)*(I.xy*wAvg.x + I.yy*wAvg.y + I.yz*wAvg.z) - (wAvg.y)*(I.xz*wAvg.x + I.yz*wAvg.y + I.zz*wAvg.z)))*inv_I_det_neg;


        wAvg = 0.5*(wAux + pc_i->getW_old());
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
    pc_i->setWX(wNew.x);
    pc_i->setWY(wNew.y);
    pc_i->setWZ(wNew.z);

    pc_i->setIXX(Iaux6.xx);
    pc_i->setIYY(Iaux6.yy);
    pc_i->setIZZ(Iaux6.zz);
    pc_i->setIXY(Iaux6.xy);
    pc_i->setIXZ(Iaux6.xz);
    pc_i->setIYZ(Iaux6.yz);

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

//#endif //PARTICLE_MODEL