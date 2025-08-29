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

__global__
void gpuParticleNodeMovement(
    IbmNodesSoA const particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES]
){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= particlesNodes.getNumNodes())
        return;

    const ParticleCenter pc = particleCenters[particlesNodes.getParticleCenterIdx()[i]];

    if(!pc.getMovable())
        return;

    // TODO: make the calculation of w_norm along with w_avg?
    const dfloat w_norm = sqrt((pc.getWAvgX() * pc.getWAvgX()) 
        + (pc.getWAvgY() * pc.getWAvgY()) 
        + (pc.getWAvgZ() * pc.getWAvgZ()));

    if(w_norm <= 1e-8)
    {
        dfloat dx,dy,dz;
        // dfloat new_pos_x,new_pos_y,new_pos_z;

        dx = pc.getPosX() - pc.getPosOldX();
        dy = pc.getPosY() - pc.getPosOldY();
        dz = pc.getPosZ() - pc.getPosOldZ();

        #ifdef IBM_BC_X_WALL
            particlesNodes.getPos().x[i] += dx;
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            if(abs(dx) > (dfloat)(IBM_BC_X_E - IBM_BC_X_0)/2.0){
                if(pc.getPosX() < pc.getPosOldX() )
                    dx = (pc.getPosX()  + (IBM_BC_X_E - IBM_BC_X_0)) - pc.getPosOldX();
                else
                    dx = (pc.getPosX()  - (IBM_BC_X_E - IBM_BC_X_0)) - pc.getPosOldX();
            }
            particlesNodes.getPos().x[i] = IBM_BC_X_0 + std::fmod((dfloat)(particlesNodes.getPos().x[i] + dx + (IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0)),(dfloat)(IBM_BC_X_E - IBM_BC_X_0));
        #endif //IBM_BC_X_PERIODIC


        #ifdef IBM_BC_Y_WALL
            particlesNodes.getPos().y[i] += dy;
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            if(abs(dy) > (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0)/2.0){
                if(pc.getPosY() < pc.getPosOldY())
                    dy = (pc.getPosY()  + (IBM_BC_Y_E - IBM_BC_Y_0)) - pc.getPosOldY();
                else
                    dy = (pc.getPosY()  - (IBM_BC_Y_E - IBM_BC_Y_0)) - pc.getPosOldY();
            }
            particlesNodes.getPos().y[i] = IBM_BC_Y_0 + std::fmod((dfloat)(particlesNodes.getPos().y[i] + dy + (IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0)),(dfloat)(IBM_BC_Y_E - IBM_BC_Y_0));
        #endif // IBM_BC_Y_PERIODIC


        #ifdef IBM_BC_Z_WALL
            particlesNodes.getPos().z[i] += dz;
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            if(abs(dz) > (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0)/2.0){
                if(pc.getPosZ() < pc.getPosOldZ())
                    dz = (pc.getPosZ() + (IBM_BC_Z_E - IBM_BC_Z_0)) - pc.getPosOldZ();
                else
                    dz = (pc.getPosZ() - (IBM_BC_Z_E - IBM_BC_Z_0)) - pc.getPosOldZ();
            }
            particlesNodes.getPos().z[i] = IBM_BC_Z_0 + std::fmod((dfloat)(particlesNodes.getPos().z[i] + dz + (IBM_BC_Z_E - IBM_BC_Z_0-IBM_BC_Z_0)),(dfloat)(IBM_BC_Z_E - IBM_BC_Z_0));
        #endif //IBM_BC_Z_PERIODIC

        return;
    }

    // TODO: these variables are the same for every particle center, optimize it
    


    dfloat x_vec = particlesNodes.getPos().x[i] - pc.getPosOldX();
    dfloat y_vec = particlesNodes.getPos().y[i] - pc.getPosOldY();
    dfloat z_vec = particlesNodes.getPos().z[i] - pc.getPosOldZ();


    #ifdef IBM_BC_X_PERIODIC
        if(abs(x_vec) > (dfloat)(IBM_BC_X_E - IBM_BC_X_0)/2.0){
            if(particlesNodes.getPos().x[i] < pc.getPosOldX() )
                particlesNodes.getPos().x[i] += (dfloat)(IBM_BC_X_E - IBM_BC_X_0) ;
            else
                particlesNodes.getPos().x[i] -= (dfloat)(IBM_BC_X_E - IBM_BC_X_0) ;
        }

        x_vec = particlesNodes.getPos().x[i] - pc.getPosOldX();
    #endif //IBM_BC_X_PERIODIC


    #ifdef IBM_BC_Y_PERIODIC
        if(abs(y_vec) > (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0)/2.0){
            if(particlesNodes.getPos().y[i] < pc.getPosOldY())
                particlesNodes.getPos().y[i] += (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0) ;
            else
                particlesNodes.getPos().y[i] -= (dfloat)(IBM_BC_Y_E - IBM_BC_Y_0) ;
        }

        y_vec = particlesNodes.getPos().y[i] - pc.getPosOldY();
    #endif //IBM_BC_Y_PERIODIC


    #ifdef IBM_BC_Z_PERIODIC
        if(abs(z_vec) > (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0)/2.0){
            if(particlesNodes.getPos().z[i] < pc.getPosOldZ())
                particlesNodes.getPos().z[i] += (dfloat)(IBM_BC_Z_E - IBM_BC_Z_0) ;
            else
                particlesNodes.getPos().z[i] -= (IBM_BC_Z_E - IBM_BC_Z_0) ;
        }

        z_vec = particlesNodes.getPos().z[i] - pc.getPosOldZ();
    #endif //IBM_BC_Z_PERIODIC

    
    
    

    const dfloat q0 = cos(0.5*w_norm);
    const dfloat qi = (pc.getWAvgX()/w_norm) * sin (0.5*w_norm);
    const dfloat qj = (pc.getWAvgY()/w_norm) * sin (0.5*w_norm);
    const dfloat qk = (pc.getWAvgZ()/w_norm) * sin (0.5*w_norm);

    const dfloat tq0m1 = (q0*q0) - 0.5;
    
    dfloat new_pos_x = pc.getPosX() + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    dfloat new_pos_y = pc.getPosY() + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    dfloat new_pos_z = pc.getPosZ() + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);

    #ifdef  IBM_BC_X_WALL
    particlesNodes.pos.x[i] =  new_pos_x;
    #endif //IBM_BC_X_WALL
    #ifdef  IBM_BC_Y_WALL
    particlesNodes.getPos().y[i] =  new_pos_y;
    #endif //IBM_BC_Y_WALL
    #ifdef  IBM_BC_Z_WALL
    particlesNodes.pos.z[i] =  new_pos_z;
    #endif //IBM_BC_Z_WALL



    #ifdef  IBM_BC_X_PERIODIC
    particlesNodes.getPos().x[i] =  IBM_BC_X_0 + std::fmod((dfloat)(new_pos_x + IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0),(dfloat)(IBM_BC_X_E - IBM_BC_X_0));
    #endif //IBM_BC_X_PERIODIC
    #ifdef  IBM_BC_Y_PERIODIC
    particlesNodes.pos.y[i] =  IBM_BC_Y_0 + std::fmod((dfloat)(new_pos_y + IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0),(dfloat)(IBM_BC_Y_E - IBM_BC_Y_0));
    #endif //IBM_BC_Y_PERIODIC
    #ifdef  IBM_BC_Z_PERIODIC
    particlesNodes.getPos().z[i] =  IBM_BC_Z_0 + std::fmod((dfloat)(new_pos_z + IBM_BC_Z_E - IBM_BC_Z_0-IBM_BC_Z_0),(dfloat)(IBM_BC_Z_E - IBM_BC_Z_0));
    #endif //IBM_BC_Z_PERIODIC
}

__global__
void gpuForceInterpolationSpread(
    IbmNodesSoA particlesNodes,
    ParticleCenter particleCenters[NUM_PARTICLES],
    dfloat *fMom,
    IbmMacrsAux ibmMacrsAux,
    const int n_gpu)
{
    // TODO: update atomic double add to use only if is double
    const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    // Shared memory to sum particles values to particle center
    // __shared__ dfloat3 sumPC[2][64];

    if (i >= particlesNodes.getNumNodes())
        return;

    dfloat aux, aux1; // aux variable for many things
    size_t idx; // index for many things

    const dfloat xIBM = particlesNodes.getPos().x[i];
    const dfloat yIBM = particlesNodes.getPos().y[i]; 
    const dfloat zIBM = particlesNodes.getPos().z[i];
    const dfloat pos[3] = {xIBM, yIBM, zIBM};

    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];
    // Base position for every index (leftest in x)

   // Base position is memory, so it discount the nodes in Z in others gpus
   /*const int posBase[3] = {
       (int)(xIBM+0.5)-P_DIST+1, 
       (int)(yIBM+0.5)-P_DIST+1, 
       (int)(zIBM+0.5)-P_DIST+1-n_gpu*NZ}
   ;*/
    /*int posBase[3] = { 
        int(xIBM - P_DIST + 0.5 - (xIBM < 1.0)), 
        int(yIBM - P_DIST + 0.5 - (yIBM < 1.0)), 
        int(zIBM - P_DIST + 0.5 - (zIBM < 1.0)) - NZ*n_gpu 
    };*/
    

    const int posBase[3] = { 
        int(xIBM) - (P_DIST) + 1, 
        int(yIBM) - (P_DIST) + 1, 
        int(zIBM) - (P_DIST) + 1 - NZ*n_gpu 
    };
    // Maximum position to interpolate in Z, used for maxIdx in Z
    int zMaxIdxPos = (n_gpu == N_GPUS-1 ? NZ : NZ+MACR_BORDER_NODES);
    // Minimum position to interpolate in Z, used for minIdx in Z
    int zMinIdxPos = (n_gpu == 0 ? 0 : -MACR_BORDER_NODES);
    #ifdef IBM_BC_Z_PERIODIC
        zMinIdxPos = -MACR_BORDER_NODES;
        zMaxIdxPos = NZ+MACR_BORDER_NODES;
    #endif
    // Maximum stencil index for each direction xyz ("index" to stop)
    const int maxIdx[3] = {
        #ifdef IBM_BC_X_WALL
            ((posBase[0]+P_DIST*2-1) < (int)NX)? P_DIST*2-1 : ((int)NX-1-posBase[0])
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            P_DIST*2-1
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL 
             ((posBase[1]+P_DIST*2-1) < (int)NY)? P_DIST*2-1 : ((int)NY-1-posBase[1])
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            P_DIST*2-1
        #endif //IBM_BC_Y_PERIODIC
        , 
        #ifdef IBM_BC_Z_WALL 
            ((posBase[2]+P_DIST*2-1) < zMaxIdxPos)? P_DIST*2-1 : ((int)zMaxIdxPos-1-posBase[2])
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            P_DIST*2-1
        #endif //IBM_BC_Z_PERIODIC
    };
    // Minimum stencil index for each direction xyz ("index" to start)
    const int minIdx[3] = {
        #ifdef IBM_BC_X_WALL
            (posBase[0] >= 0)? 0 : -posBase[0]
        #endif //IBM_BC_X_WALL
        #ifdef IBM_BC_X_PERIODIC
            0
        #endif //IBM_BC_X_PERIODIC
        ,
        #ifdef IBM_BC_Y_WALL 
            (posBase[1] >= 0)? 0 : -posBase[1]
        #endif //IBM_BC_Y_WALL
        #ifdef IBM_BC_Y_PERIODIC
            0
        #endif //IBM_BC_Y_PERIODIC
        , 
        #ifdef IBM_BC_Z_WALL 
            (posBase[2] >= zMinIdxPos)? 0 : zMinIdxPos-posBase[2]
        #endif //IBM_BC_Z_WALL
        #ifdef IBM_BC_Z_PERIODIC
            0
        #endif //IBM_BC_Z_PERIODIC
    };


    // Particle stencil out of the domain
    if(maxIdx[0] <= 0 || maxIdx[1] <= 0 || maxIdx[2] <= 0)
        return;
    // Particle stencil out of the domain
    if(minIdx[0] >= P_DIST*2 || minIdx[1] >= P_DIST*2 || minIdx[2] >= P_DIST*2)
        return;

    // #ifdef EXTERNAL_DUCT_BC
    //     if (dfloat((xIBM - 0.5*NX + 0.5)*(xIBM - 0.5*NX + 0.5)+(yIBM - 0.5*NY + 0.5)*(yIBM - 0.5*NY + 0.5))>= dfloat((EXTERNAL_DUCT_BC_RADIUS)*(EXTERNAL_DUCT_BC_RADIUS)))
    //         return;       
    // #endif    


    for(int i = 0; i < 3; i++){
        for(int j=minIdx[i]; j <= maxIdx[i]; j++){
            stencilVal[i][j] = stencil(posBase[i]+j-(pos[i]-(i == 2? NZ*n_gpu : 0)));
        }
    }

    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    // Interpolation (zyx for memory locality)
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];
                // same as aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);


                idx = idxScalarWBorder(
                    #ifdef IBM_BC_X_WALL
                        posBase[0]+xi
                    #endif //IBM_BC_X_WALL
                    #ifdef IBM_BC_X_PERIODIC
                        IBM_BC_X_0 + (posBase[0]+xi + IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0)%(IBM_BC_X_E - IBM_BC_X_0)
                    #endif //IBM_BC_X_PERIODIC
                    ,
                    #ifdef IBM_BC_Y_WALL 
                        posBase[1]+yj
                    #endif //IBM_BC_Y_WALL
                    #ifdef IBM_BC_Y_PERIODIC    
                        IBM_BC_Y_0 + (posBase[1]+yj + IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0)%(IBM_BC_Y_E - IBM_BC_Y_0)
                    #endif //IBM_BC_Y_PERIODIC
                    , 
                    #ifdef IBM_BC_Z_WALL  // +MACR_BORDER_NODES in z because of the ghost nodes
                        posBase[2]+zk
                    #endif //IBM_BC_Z_WALL
                    #ifdef IBM_BC_Z_PERIODIC
                        posBase[2]+zk
                    #endif //IBM_BC_Z_PERIODIC
                );
                // rhoVar += macr.rho[idx] * aux;
                // uxVar += macr.u.x[idx] * aux;
                // uyVar += macr.u.y[idx] * aux;
                // uzVar += macr.u.z[idx] * aux;
            }
        }
    }

    // Index of particle center of this particle node
    idx = particlesNodes.getParticleCenterIdx()[i];

    // Velocity on node given the particle velocity and rotation
    dfloat ux_calc = 0;
    dfloat uy_calc = 0;
    dfloat uz_calc = 0;

    // Load position of particle center
    const dfloat x_pc = particleCenters[idx].getPosX();
    const dfloat y_pc = particleCenters[idx].getPosY();
    const dfloat z_pc = particleCenters[idx].getPosZ();

    dfloat dx = xIBM - x_pc;
    dfloat dy = yIBM - y_pc;
    dfloat dz = zIBM - z_pc;

    #ifdef IBM_BC_X_PERIODIC
    if(abs(dx) > (dfloat)((IBM_BC_X_E - IBM_BC_X_0))/2.0){
        if(dx < 0)
            dx = (xIBM + (IBM_BC_X_E - IBM_BC_X_0)) - x_pc;
        else
            dx = (xIBM - (IBM_BC_X_E - IBM_BC_X_0)) - x_pc;
    }
    #endif //IBM_BC_X_PERIODIC
    
    #ifdef IBM_BC_Y_PERIODIC
    if(abs(dy) > (dfloat)((IBM_BC_Y_E - IBM_BC_Y_0))/2.0){
        if(dy < 0)
            dy = (yIBM + (IBM_BC_Y_E - IBM_BC_Y_0)) - y_pc;
        else
            dy = (yIBM - (IBM_BC_Y_E - IBM_BC_Y_0)) - y_pc;
    }
    #endif //IBM_BC_Y_PERIODIC

    #ifdef IBM_BC_Z_PERIODIC
    if(abs(dz) > (dfloat)((IBM_BC_Z_E - IBM_BC_Z_0))/2.0){
        if(dz < 0)
            dz = (zIBM + (IBM_BC_Z_E - IBM_BC_Z_0)) - z_pc;
        else
            dz = (zIBM - (IBM_BC_Z_E - IBM_BC_Z_0)) - z_pc;
    }
    #endif //IBM_BC_Z_PERIODIC

    // Calculate velocity on node if particle is movable
    if(particleCenters[idx].getMovable()){
        // Load velocity and rotation velocity of particle center
        const dfloat vx_pc = particleCenters[idx].getVelX();
        const dfloat vy_pc = particleCenters[idx].getVelY();
        const dfloat vz_pc = particleCenters[idx].getVelZ();

        const dfloat wx_pc = particleCenters[idx].getWX();
        const dfloat wy_pc = particleCenters[idx].getWY();
        const dfloat wz_pc = particleCenters[idx].getWZ();

        // velocity on node, given the center velocity and rotation
        // (i.e. no slip boundary condition velocity)
        ux_calc = vx_pc + (wy_pc * (dz) - wz_pc * (dy));
        uy_calc = vy_pc + (wz_pc * (dx) - wx_pc * (dz));
        uz_calc = vz_pc + (wx_pc * (dy) - wy_pc * (dx));
    }

    const dfloat dA = particlesNodes.getS()[i];
    aux = 2 * rhoVar * dA * IBM_THICKNESS;

    dfloat3 deltaF;
    deltaF.x = aux * (uxVar - ux_calc);
    deltaF.y = aux * (uyVar - uy_calc);
    deltaF.z = aux * (uzVar - uz_calc);

    // Calculate IBM forces
    const dfloat fxIBM = particlesNodes.getF().x[i] + deltaF.x;
    const dfloat fyIBM = particlesNodes.getF().y[i] + deltaF.y;
    const dfloat fzIBM = particlesNodes.getF().z[i] + deltaF.z;

    // Spreading (zyx for memory locality)
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];
                // same as aux = stencil(x - xIBM) * stencil(y - yIBM) * stencil(z - zIBM);
 
                idx = idxScalarWBorder(
                    #ifdef IBM_BC_X_WALL
                        posBase[0]+xi
                    #endif //IBM_BC_X_WALL
                    #ifdef IBM_BC_X_PERIODIC
                        IBM_BC_X_0 + (posBase[0]+xi + IBM_BC_X_E - IBM_BC_X_0-IBM_BC_X_0)%(IBM_BC_X_E - IBM_BC_X_0)
                    #endif //IBM_BC_X_PERIODIC
                    ,
                    #ifdef IBM_BC_Y_WALL 
                        posBase[1]+yj
                    #endif //IBM_BC_Y_WALL
                    #ifdef IBM_BC_Y_PERIODIC
                        IBM_BC_Y_0 + (posBase[1]+yj + IBM_BC_Y_E - IBM_BC_Y_0-IBM_BC_Y_0)%(IBM_BC_Y_E - IBM_BC_Y_0)
                    #endif //IBM_BC_Y_PERIODIC
                    , 
                    #ifdef IBM_BC_Z_WALL  // +MACR_BORDER_NODES in z because of the ghost nodes
                        posBase[2]+zk
                    #endif //IBM_BC_Z_WALL
                    #ifdef IBM_BC_Z_PERIODIC
                        posBase[2]+zk
                        //OLD: IBM_BC_Z_0 + (posBase[2]+zk+ (IBM_BC_Z_E-n_gpu*NZ) - IBM_BC_Z_0-IBM_BC_Z_0)%((IBM_BC_Z_E-n_gpu*NZ) - IBM_BC_Z_0)
                    #endif //IBM_BC_Z_PERIODIC
                );

                #ifdef EXTERNAL_DUCT_BC
                dfloat xCenter = DUCT_CENTER_X;
                dfloat yCenter = DUCT_CENTER_Y;

                // int n_gpu2 = ((int)((posBase[2]+zk)/NZ) + 100*N_GPUS)%N_GPUS;

                dfloat pos_r_i = sqrt((posBase[0] + xi - xCenter)*(posBase[0] + xi - xCenter) + (posBase[1] + yj - yCenter)*(posBase[1] + yj - yCenter));
                //if point is outside of the duct is not computed
                if (pos_r_i < EXTERNAL_DUCT_BC_RADIUS)
                {
                    atomicAdd(&(ibmMacrsAux.getFAux(n_gpu).x[idx]), -deltaF.x * aux);
                    atomicAdd(&(ibmMacrsAux.getFAux(n_gpu).y[idx]), -deltaF.y * aux);
                    atomicAdd(&(ibmMacrsAux.getFAux(n_gpu).z[idx]), -deltaF.z * aux);

                    // Update velocities field
                    const dfloat inv_rho = 1 / macr.rho[idx];
                    atomicAdd(&(ibmMacrsAux.getVelAux(n_gpu).x[idx]), 0.5 * -deltaF.x * aux * inv_rho);
                    atomicAdd(&(ibmMacrsAux.getVelAux(n_gpu).y[idx]), 0.5 * -deltaF.y * aux * inv_rho);
                    atomicAdd(&(ibmMacrsAux.getVelAux(n_gpu).z[idx]), 0.5 * -deltaF.z * aux * inv_rho);

                    // if (posBase[2]+zk < NZ)
                    // {
                    //atomicAdd(&(macr.pbound[idx]), aux);
                    // }
                }
                #endif
                #ifndef EXTERNAL_DUCT_BC
                atomicAdd(&(ibmMacrsAux.getFAux(n_gpu).x[idx]), -deltaF.x * aux);
                atomicAdd(&(ibmMacrsAux.getFAux(n_gpu).y[idx]), -deltaF.y * aux);
                atomicAdd(&(ibmMacrsAux.getFAux(n_gpu).z[idx]), -deltaF.z * aux);

                // Update velocities field
                // const dfloat inv_rho = 1 / macr.rho[idx];
                // atomicAdd(&(ibmMacrsAux.getVelAux(n_gpu).x[idx]), 0.5 * -deltaF.x * aux * inv_rho);
                // atomicAdd(&(ibmMacrsAux.getVelAux(n_gpu).y[idx]), 0.5 * -deltaF.y * aux * inv_rho);
                // atomicAdd(&(ibmMacrsAux.getVelAux(n_gpu).z[idx]), 0.5 * -deltaF.z * aux * inv_rho);
            #endif 

            }
        }
    }

    // Update node force
    particlesNodes.getF().x[i] = fxIBM;
    particlesNodes.getF().y[i] = fyIBM;
    particlesNodes.getF().z[i] = fzIBM;

    // Update node delta force
    particlesNodes.getDeltaF().x[i] = deltaF.x;
    particlesNodes.getDeltaF().y[i] = deltaF.y;
    particlesNodes.getDeltaF().z[i] = deltaF.z;

    // Particle node delta momentum
    idx = particlesNodes.getParticleCenterIdx()[i];

    const dfloat3 deltaMomentum = dfloat3(
        (dy) * deltaF.z - (dz) * deltaF.y,
        (dz) * deltaF.x - (dx) * deltaF.z,
        (dx) * deltaF.y - (dy) * deltaF.x
    );
    // Avaliar a possibilidade de separar a soma da força e momentos em um kernel diferente
    atomicAdd(&(particleCenters[idx].getFXatomic()), deltaF.x);
    atomicAdd(&(particleCenters[idx].getFYatomic()), deltaF.y);
    atomicAdd(&(particleCenters[idx].getFZatomic()), deltaF.z);

    atomicAdd(&(particleCenters[idx].getMXatomic()), deltaMomentum.x);
    atomicAdd(&(particleCenters[idx].getMYatomic()), deltaMomentum.y);
    atomicAdd(&(particleCenters[idx].getMZatomic()), deltaMomentum.z);
}


void ibmSimulation(
    ParticlesSoA particles,
    IbmMacrsAux ibmMacrsAux,
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
       // int nxt = (i+1) % N_GPUS;
        // Copy macroscopics
       // gpuCopyBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[i]>>>(macr[i], macr[nxt]); Verificar se é necessário
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
 
     for (int i = 0; i < IBM_MAX_ITERATION; i++)
     {
         for(int j = 0; j < N_GPUS; j++){
             // If GPU has nodes in it
             if(particles.getNodesSoA()[j].getNumNodes() > 0){
                 checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
                 // Make the interpolation of LBM and spreading of IBM forces
                 gpuForceInterpolationSpread<<<gridNodesIBM[j], threadsNodesIBM[j], 
                     0, streamParticles>>>(
                        particles.getNodesSoA()[i], particles.getPCenterArray(), &fMom[0], ibmMacrsAux, j);
                 checkCudaErrors(cudaStreamSynchronize(streamParticles));
                 getLastCudaError("IBM interpolation spread error\n");
             }
         }
 
         checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
         // Update particle velocity using body center force and constant forces
         // Migrar
         gpuUpdateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(
            particles.getPCenterArray());
         checkCudaErrors(cudaStreamSynchronize(streamParticles));
         getLastCudaError("IBM update particle center velocity error\n");
 
         // Sum border macroscopics
         // for(int j = 0; j < N_GPUS; j++){
         //     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
         //     int nxt = (j+1) % N_GPUS;
         //     int prv = (j-1+N_GPUS) % N_GPUS;
         //     bool run_nxt = nxt != 0;
         //     bool run_prv = prv != (N_GPUS-1);
         //     #ifdef IBM_BC_Z_PERIODIC
         //     run_nxt = true;
         //     run_prv = true;
         //     #endif
             
         //     if(run_nxt){
         //         gpuSumBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[nxt], ibmMacrsAux, j, 1);
         //         checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
         //     }
         //     if(run_prv){
         //         gpuSumBorderMacr<<<copyMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[prv], ibmMacrsAux, j, -1);
         //         checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
         //     }
         //     getLastCudaError("Sum border macroscopics error\n");
         // }
 
         // #if IBM_EULER_OPTIMIZATION
 
         // for(int j = 0; j < N_GPUS; j++){
         //     if(pEulerNodes->currEulerNodes[j] > 0){
         //         checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
         //         dim3 currGrid(pEulerNodes->currEulerNodes[j]/64+(pEulerNodes->currEulerNodes[j]%64? 1 : 0), 1, 1);
         //         gpuEulerSumIBMAuxsReset<<<currGrid, 64, 0, streamLBM[j]>>>(macr[j], ibmMacrsAux,
         //             pEulerNodes->eulerIndexesUpdate[j], pEulerNodes->currEulerNodes[j], j);
         //         checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
         //         getLastCudaError("IBM sum auxiliary values error\n");
         //     }
         // }
         // #else
         // for(int j = 0; j < N_GPUS; j++){
         //     checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
         //     gpuEulerSumIBMAuxsReset<<<borderMacrGrid, threadsLBM, 0, streamLBM[j]>>>(macr[j], ibmMacrsAux, j);
         //     checkCudaErrors(cudaStreamSynchronize(streamLBM[j]));
         // }
         // #endif
 
     }

    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    // Update particle center position and its old values
    gpuParticleMovement<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(
        particles.getPCenterArray());
    checkCudaErrors(cudaStreamSynchronize(streamParticles));
    getLastCudaError("IBM particle movement error\n");

    for(int i = 0; i < N_GPUS; i++){
        // If GPU has nodes in it
        if(particles.getNodesSoA()[i].getNumNodes() > 0){ // particles.nodesSoA[i].numNodes
            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
            // Update particle nodes positions
            gpuParticleNodeMovement<<<gridNodesIBM[i], threadsNodesIBM[i], 0, streamParticles>>>(
                particles.getNodesSoA()[i], particles.getPCenterArray());
            checkCudaErrors(cudaStreamSynchronize(streamParticles));
            getLastCudaError("IBM particle movement error\n");
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
 
}

// #endif //PARTICLE_MODEL