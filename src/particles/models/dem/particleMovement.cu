

//functions related to the rigid body body of the particle and discretization

#include "particleMovement.cuh"

#ifdef PARTICLE_MODEL
__global__
void gpuUpdateParticleOldValues(
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,    
    unsigned int step)
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
    //pc_i->setDPInternalX(RHO_0 * pc_i->getVolume() * (pc_i->getVelX() - pc_i->getVelOldX())); //;
    //pc_i->setDPInternalY(RHO_0 * pc_i->getVolume() * (pc_i->getVelY() - pc_i->getVelOldY())); //;
    //pc_i->setDPInternalZ(RHO_0 * pc_i->getVolume() * (pc_i->getVelZ() - pc_i->getVelOldZ())); //;
    pc_i->setDPInternalX(0.0);
    pc_i->setDPInternalY(0.0);
    pc_i->setDPInternalZ(0.0);

    // Internal angular momentum delta = (rho_f/rho_p)*I*delta(omega)/delta(t)
    // https://doi.org/10.1016/j.compfluid.2011.05.011
    
    //pc_i->setDLInternalX((RHO_0 / pc_i->getDensity()) * pc_i->getIXX() * (pc_i->getWX() - pc_i->getWOldX())); 
    //pc_i->setDLInternalY((RHO_0 / pc_i->getDensity()) * pc_i->getIYY() * (pc_i->getWY() - pc_i->getWOldY())); 
    //pc_i->setDLInternalZ((RHO_0 / pc_i->getDensity()) * pc_i->getIZZ() * (pc_i->getWZ() - pc_i->getWOldZ())); 
    pc_i->setDLInternalX(0.0);
    pc_i->setDLInternalY(0.0);
    pc_i->setDLInternalZ(0.0);

    #ifdef PARTICLE_DEBUG
    printf("gpuUpdateParticleOldValues 2 pos  x: %e y: %e z: %e\n",pc_i->getPosOldX(),pc_i->getPosOldY(),pc_i->getPosOldZ());
    printf("gpuUpdateParticleOldValues 3 pos  x: %e y: %e z: %e\n",pc_i->getVelOldX(),pc_i->getVelOldY(),pc_i->getVelOldZ());
    printf("gpuUpdateParticleOldValues 4 pos  x: %e y: %e z: %e\n",pc_i->getWOldX(),pc_i->getWOldY(),pc_i->getWOldZ());
    printf("gpuUpdateParticleOldValues 5 pos  x: %e y: %e z: %e\n",pc_i->getFOldX(),pc_i->getFOldY(),pc_i->getFOldZ());
    printf("gpuUpdateParticleOldValues 6 pos  x: %e y: %e z: %e\n",pc_i->getFX(),pc_i->getFY(),pc_i->getFZ());
    printf("gpuUpdateParticleOldValues 7 pos  x: %e y: %e z: %e\n",pc_i->getMX(),pc_i->getMY(),pc_i->getMZ());
    #endif //PARTICLE_DEBUG

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
    int lastIndex,    
    unsigned int step)
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

    #ifdef PARTICLE_DEBUG
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 f  x: %e y: %e z: %e\n",pc_i->getF().x,pc_i->getF().y,pc_i->getF().z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 m  x: %e y: %e z: %e\n",pc_i->getMX(),pc_i->getMY(),pc_i->getMZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 DP  x: %e y: %e z: %e\n",pc_i->getDP_internal().x,pc_i->getDP_internal().y,pc_i->getDP_internal().z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 pos_old  x: %e y: %e z: %e\n",pc_i->getPosOldX(),pc_i->getPosOldY(),pc_i->getPosOldZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 vel_old  x: %e y: %e z: %e\n",pc_i->getVelOldX(),pc_i->getVelOldY(),pc_i->getVelOldZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 w_old  x: %e y: %e z: %e\n",pc_i->getWOldX(),pc_i->getWOldY(),pc_i->getWOldZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 f_old  x: %e y: %e z: %e\n",pc_i->getFOldX(),pc_i->getFOldY(),pc_i->getFOldZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 m_old  x: %e y: %e z: %e\n",pc_i->getMOldX(),pc_i->getMOldY(),pc_i->getMOldZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 volume %e\n",pc_i->getVolume());
    printf("gpuUpdateParticleCenterVelocityAndRotation 1 density %e\n",pc_i->getDensity());
    #endif //PARTICLE_DEBUG

    // Update particle center velocity using its surface forces and the body forces
    dfloat3 g = {GX,GY,GZ};
    const dfloat inv_volume = 1 / pc_i->getVolume();
    pc_i->setVel(pc_i->getVel_old() + (((pc_i->getF_old() + pc_i->getF())/2 + pc_i->getDP_internal())*inv_volume
                + (pc_i->getDensity() - FLUID_DENSITY)*g) / (pc_i->getDensity()));
    //pc_i->setVel(pc_i->getVel_old() + (((pc_i->getF_old() + pc_i->getF())/2 + pc_i->getDP_internal())) / (pc_i->getVolume()) 
    //            + (1.0 - FLUID_DENSITY/pc_i->getDensity()) * g);


    // Update particle angular velocity  

    dfloat6 I = pc_i->getI();
    dfloat inv_I_det_neg = 1.0/(I.zz*I.xy*I.xy + I.yy*I.xz*I.xz + I.xx*I.yz*I.yz - I.xx*I.yy*I.zz - 2*I.xy*I.xz*I.yz);
    dfloat3 wAux = pc_i->getW_old();
    dfloat3 wAvg = (pc_i->getW_old() + pc_i->getW())/2;
    dfloat3 LM_avg = pc_i->getDL_internal() + (pc_i->getM_old() + pc_i->getM())/2;

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


        wAvg = (wAux + pc_i->getW_old())/2;
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

    #ifdef PARTICLE_DEBUG
    printf("gpuUpdateParticleCenterVelocityAndRotation 2 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("gpuUpdateParticleCenterVelocityAndRotation 2 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("gpuUpdateParticleCenterVelocityAndRotation 2 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    #endif //PARTICLE_DEBUG
}

__global__
void gpuParticleMovement(
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex,    
    unsigned int step)
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

    #ifdef PARTICLE_DEBUG
    printf("gpuParticleMovement 1 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("gpuParticleMovement 1 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("gpuParticleMovement 1 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    #endif //PARTICLE_DEBUG

    #ifdef BC_X_WALL
        pc_i->setPosX(pc_i->getPosX() + (pc_i->getVelX() + pc_i->getVelOldX())/2);
    #endif //BC_X_WALL
    #ifdef BC_X_PERIODIC
        dfloat dx  = (pc_i->getVelX() + pc_i->getVelOldX())/2;
        pc_i->setPosX(std::fmod((dfloat)(pc_i->getPosX() + dx + NX),(dfloat)(NX)));
    #endif //BC_X_PERIODIC

    #ifdef BC_Y_WALL
        pc_i->setPosY(pc_i->getPosY() + (pc_i->getVelY() + pc_i->getVelOldY())/2);
    #endif //BC_Y_WALL
    #ifdef BC_Y_PERIODIC
        dfloat dy  = (pc_i->getVelY() + pc_i->getVelOldY())/2;
        pc_i->setPosY(std::fmod((dfloat)(pc_i->getPosY() + dy + NY),(dfloat)(NY)));
    #endif //BC_Y_PERIODIC

    #ifdef BC_Z_WALL
        pc_i->setPosZ(pc_i->getPosZ() + (pc_i->getVelZ() + pc_i->getVelOldZ())/2);
    #endif //BC_Z_WALL
    #ifdef BC_Z_PERIODIC
        dfloat dz  = (pc_i->getVelZ() + pc_i->getVelOldZ())/2;
        pc_i->setPosZ(std::fmod((dfloat)(pc_i->getPosZ() + dz + NZ_TOTAL),(dfloat)(NZ_TOTAL)));
    #endif //BC_Z_PERIODIC

    //Compute angular velocity
    pc_i->setW_avg((pc_i->getW() + pc_i->getW_old())/2);

    //update angular position
    pc_i->setW_pos(pc_i->getW_pos() + pc_i->getW_avg());

    #ifdef PARTICLE_DEBUG
    printf("gpuParticleMovement 2 pos  x: %e y: %e z: %e\n",pc_i->getPosX(),pc_i->getPosY(),pc_i->getPosZ());
    printf("gpuParticleMovement 2 vel  x: %e y: %e z: %e\n",pc_i->getVel().x,pc_i->getVel().y,pc_i->getVel().z);
    printf("gpuParticleMovement 2 w  x: %e y: %e z: %e\n",pc_i->getWX(),pc_i->getWY(),pc_i->getWZ());
    #endif //PARTICLE_DEBUG


    //compute orientation vector
    const dfloat w_norm = sqrt((pc_i->getWAvgX() * pc_i->getWAvgX()) 
                             + (pc_i->getWAvgY() * pc_i->getWAvgY()) 
                             + (pc_i->getWAvgZ() * pc_i->getWAvgZ()));

    const dfloat q0 = cos(w_norm/2);
    const dfloat qi = (pc_i->getWAvgX()/w_norm) * sin (w_norm/2);
    const dfloat qj = (pc_i->getWAvgY()/w_norm) * sin (w_norm/2);
    const dfloat qk = (pc_i->getWAvgZ()/w_norm) * sin (w_norm/2);
    const dfloat tq0m1 = (q0*q0) - 0.5;

    dfloat x_vec = pc_i->getSemiAxis1X() - pc_i->getPosOldX();
    dfloat y_vec = pc_i->getSemiAxis1Y() - pc_i->getPosOldY();
    dfloat z_vec = pc_i->getSemiAxis1Z() - pc_i->getPosOldZ();

    //update semiaxis position
    pc_i->setSemiAxis1X(pc_i->getPosX() + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec));
    pc_i->setSemiAxis1Y(pc_i->getPosY() + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec));
    pc_i->setSemiAxis1Z(pc_i->getPosZ() + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec));

    x_vec = pc_i->getSemiAxis2X() - pc_i->getPosOldX();
    y_vec = pc_i->getSemiAxis2Y() - pc_i->getPosOldY();
    z_vec = pc_i->getSemiAxis2Z() - pc_i->getPosOldZ();

    pc_i->setSemiAxis2X(pc_i->getPosX() +  2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec));
    pc_i->setSemiAxis2Y(pc_i->getPosY() +  2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1  + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec));
    pc_i->setSemiAxis2Z(pc_i->getPosZ() +  2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1  + (qk*qk))*z_vec));

    x_vec = pc_i->getSemiAxis3X() - pc_i->getPosOldX();
    y_vec = pc_i->getSemiAxis3Y() - pc_i->getPosOldY();
    z_vec = pc_i->getSemiAxis3Z() - pc_i->getPosOldZ();

    pc_i->setSemiAxis3X(pc_i->getPosX() +  2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec));
    pc_i->setSemiAxis3X(pc_i->getPosY() +  2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1  + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec));
    pc_i->setSemiAxis3X(pc_i->getPosZ() +  2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1  + (qk*qk))*z_vec));

}

#endif //PARTICLE_MODEL