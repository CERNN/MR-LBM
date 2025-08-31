// #ifdef PARTICLE_MODEL

#include "ibm.cuh"


void ibmSimulation(
    ParticlesSoA* particles,
    IbmMacrsAux ibmMacrsAux,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
){

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    MethodRange range = particles->getMethodRange(IBM);

    int numIBMParticles = range.last - range.first + 1; //printf("number of ibm particles %d \n",numIBMParticles);
    const unsigned int threadsNodesIBM = 64;
    unsigned int pNumNodes = particles->getNodesSoA()->getNumNodes();   //printf("Number of IBM %d\n",pNumNodes);
    const unsigned int gridNodesIBM = pNumNodes % threadsNodesIBM ? pNumNodes / threadsNodesIBM + 1 : pNumNodes / threadsNodesIBM;

    if (particles == nullptr) {
        printf("Error: particles is nullptr\n");
        return;
    }

    checkCudaErrors(cudaStreamSynchronize(streamParticles));

    if (range.first < 0 || range.last >= NUM_PARTICLES || range.first > range.last) {
    printf("Error: Invalid range - first: %d, last: %d, NUM_PARTICLES: %d\n", 
            range.first, range.last, NUM_PARTICLES);
    return;
    }

    ParticleCenter* pArray = particles->getPCenterArray();


    //printf("Inside ibmSimulation \n");

    // Update particle center position and its old values
    gpuUpdateParticleOldValues<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(pArray,range.first,range.last);
    checkCudaErrors(cudaStreamSynchronize(streamParticles));

    // Reset forces in all IBM nodes;
    IbmNodesSoA h_nodes = *(particles->getNodesSoA());
    IbmNodesSoA* d_nodes = &h_nodes;
    cudaMalloc(&d_nodes, sizeof(IbmNodesSoA));
    cudaMemcpy(d_nodes, &h_nodes, sizeof(IbmNodesSoA), cudaMemcpyHostToDevice);
    if(pNumNodes > 0){
        gpuResetNodesForces<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(d_nodes);
        checkCudaErrors(cudaStreamSynchronize(streamParticles));
        getLastCudaError("Reset IBM nodes forces error\n");
    }
/*
    // Calculate collision force between particles
    //gpuParticlesCollisionHandler<<<GRID_PCOLLISION_IBM, THREADS_PCOLLISION_IBM, 0, streamIBM[0]>>>(particles.pCenterArray,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles)); 

    // First update particle velocity using body center force and constant forces
    gpuUpdateParticleCenterVelocityAndRotation <<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles >>>(pArray,range.first,range.last);
    checkCudaErrors(cudaStreamSynchronize(streamParticles));
    getLastCudaError("IBM update particle center velocity error\n");


    for (int i = 0; i < IBM_MAX_ITERATION; i++)
    {
        for(int j = 0; j < N_GPUS; j++){
            // If GPU has nodes in it
            if(particles.getNodesSoA()[j].getNumNodes() > 0){
                checkCudaErrors(cudaSetDevice(GPUS_TO_USE[j]));
                // Make the interpolation of LBM and spreading of IBM forces
                //gpuForceInterpolationSpread<<<gridNodesIBM, threadsNodesIBM,0, streamParticles>>>(particles.getNodesSoA()[i], particles.getPCenterArray(), &fMom[0], j);
                checkCudaErrors(cudaStreamSynchronize(streamParticles));
                getLastCudaError("IBM interpolation spread error\n");
             }
         }
 
        // Update particle velocity using body center force and constant forces
        gpuUpdateParticleCenterVelocityAndRotation<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(particles.getPCenterArray());
        checkCudaErrors(cudaStreamSynchronize(streamParticles));
        getLastCudaError("IBM update particle center velocity error\n");
    }

    
    // Update particle center position and its old values
    gpuParticleMovement<<<GRID_PARTICLES_IBM, THREADS_PARTICLES_IBM, 0, streamParticles>>>(pArray,range.first,range.last);
    checkCudaErrors(cudaStreamSynchronize(streamParticles));
    getLastCudaError("IBM particle movement error\n");



    if(pNumNodes > 0){
        gpuParticleNodeMovement<<<gridNodesIBM, threadsNodesIBM, 0, streamParticles>>>(particles.getNodesSoA()[i], particles.getPCenterArray());
       checkCudaErrors(cudaStreamSynchronize(streamParticles));
        getLastCudaError("IBM particle movement error\n");
    }
    checkCudaErrors(cudaDeviceSynchronize());
*/
}




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
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    int firstIndex,
    int lastIndex
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx >= particlesNodes->getNumNodes())
        return;

    const dfloat3SoA pos = particlesNodes->getPos();

    //direct copy since we are not modifying
    const ParticleCenter pc_i = pArray[particlesNodes->getParticleCenterIdx()[idx]];

    if(!pc_i.getMovable())
        return;

    // TODO: make the calculation of w_norm along with w_avg?
    const dfloat w_norm = sqrt((pc_i.getWAvgX() * pc_i.getWAvgX()) 
                             + (pc_i.getWAvgY() * pc_i.getWAvgY()) 
                             + (pc_i.getWAvgZ() * pc_i.getWAvgZ()));

    // check the norm to see if is worth computing the rotation
    if(w_norm <= 1e-8)
    {
        dfloat dx,dy,dz;
        // dfloat new_pos_x,new_pos_y,new_pos_z;

        dx = pc_i.getPosX() - pc_i.getPosOldX();
        dy = pc_i.getPosY() - pc_i.getPosOldY();
        dz = pc_i.getPosZ() - pc_i.getPosOldZ();

        
        #ifdef BC_X_WALL
            pos.x[idx] += dx;
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC
            if(abs(dx) > (dfloat)(NX)/2.0){
                if(pc_i.getPosX() < pc_i.getPosOldX() )
                    dx = (pc_i.getPosX() + NX) - pc_i.getPosOldX();
                else
                    dx = (pc_i.getPosX() - NX) - pc_i.getPosOldX();
            }
            pos.x[idx] = std::fmod((dfloat)(pos.x[idx] + dx + NX),(dfloat)(NX));
        #endif //BC_X_PERIODIC

        #ifdef BC_Y_WALL
            pos.y[idx] += dy;
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
            if(abs(dy) > (dfloat)(NY)/2.0){
                if(pc_i.getPosY() < pc_i.getPosOldY() )
                    dy = (pc_i.getPosY() + NY) - pc_i.getPosOldY();
                else
                    dy = (pc_i.getPosY() - NY) - pc_i.getPosOldY();
            }
            pos.y[idx] = std::fmod((dfloat)(pos.y[idx] + dy + NY),(dfloat)(NY));
        #endif //BC_Y_PERIODIC

        #ifdef BC_Z_WALL
            pos.z[idx] += dz;
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            if(abs(dz) > (dfloat)(NZ)/2.0){
                if(pc_i.getPosZ() < pc_i.getPosOldZ() )
                    dz = (pc_i.getPosZ() + NZ_TOTAL) - pc_i.getPosOldZ();
                else
                    dz = (pc_i.getPosZ() - NZ_TOTAL) - pc_i.getPosOldZ();
            }
            pos.z[idx] = std::fmod((dfloat)(pos.z[idx] + dz + NZ_TOTAL),(dfloat)(NZ_TOTAL));
        #endif //BC_Z_PERIODIC
        
        //ealier return since is no longer necessary
        return;
    }

    //compute vector between the node and the partic
    dfloat x_vec = pos.x[idx] - pc_i.getPosOldX();
    dfloat y_vec = pos.y[idx] - pc_i.getPosOldY();
    dfloat z_vec = pos.z[idx] - pc_i.getPosOldZ();


    #ifdef IBM_BC_X_PERIODIC
        if(abs(x_vec) > (dfloat)(NX)/2.0){
            if(pos.x[idx] < pc_i.getPosOldX() )
                pos.x[idx] += (dfloat)(NX) ;
            else
                pos.x[idx] -= (dfloat)(NX) ;
        }
        x_vec = pos.x[idx] - pc_i.getPosOldX();
    #endif //IBM_BC_X_PERIODIC


    #ifdef IBM_BC_Y_PERIODIC
        if(abs(y_vec) > (dfloat)(NY)/2.0){
            if(pos.y[idx] < pc_i.getPosOldY())
                pos.y[idx] += (dfloat)(NY) ;
            else
                pos.y[idx] -= (dfloat)(NY) ;
        }

        y_vec = pos.y[idx] - pc_i.getPosOldY();
    #endif //IBM_BC_Y_PERIODIC


    #ifdef IBM_BC_Z_PERIODIC
        if(abs(z_vec) > (dfloat)(NZ_TOTAL)/2.0){
            if(pos.z[idx] < pc_i.getPosOldZ())
                pos.z[idx] += (dfloat)(NZ_TOTAL) ;
            else
                pos.z[idx] -= (NZ_TOTAL) ;
        }

        z_vec = pos.z[idx] - pc_i.getPosOldZ();
    #endif //IBM_BC_Z_PERIODIC

       
    // compute rotation quartenion
    const dfloat q0 = cos(0.5*w_norm);
    const dfloat qi = (pc_i.getWAvgX()/w_norm) * sin (0.5*w_norm);
    const dfloat qj = (pc_i.getWAvgY()/w_norm) * sin (0.5*w_norm);
    const dfloat qk = (pc_i.getWAvgZ()/w_norm) * sin (0.5*w_norm);

    const dfloat tq0m1 = (q0*q0) - 0.5;
    
    dfloat new_pos_x = pc_i.getPosX() + 2 * (   (tq0m1 + (qi*qi))*x_vec + ((qi*qj) - (q0*qk))*y_vec + ((qi*qk) + (q0*qj))*z_vec);
    dfloat new_pos_y = pc_i.getPosY() + 2 * ( ((qi*qj) + (q0*qk))*x_vec +   (tq0m1 + (qj*qj))*y_vec + ((qj*qk) - (q0*qi))*z_vec);
    dfloat new_pos_z = pc_i.getPosZ() + 2 * ( ((qi*qj) - (q0*qj))*x_vec + ((qj*qk) + (q0*qi))*y_vec +   (tq0m1 + (qk*qk))*z_vec);

    //update node position
    #ifdef  IBM_BC_X_WALL
        pos.x[idx] =  new_pos_x;
    #endif //IBM_BC_X_WALL
    #ifdef  IBM_BC_X_PERIODIC
        pos.x[idx] =  std::fmod((dfloat)(new_pos_x + NX),(dfloat)(NX));
    #endif //IBM_BC_X_PERIODIC

    #ifdef  IBM_BC_Y_WALL
        pos.y[idx] =  new_pos_y;
    #endif //IBM_BC_Y_WALL
    #ifdef  IBM_BC_Y_PERIODIC
        pos.y[idx] = std::fmod((dfloat)(new_pos_y + NY),(dfloat)(NY));
    #endif //IBM_BC_Y_PERIODIC

    #ifdef  IBM_BC_Z_WALL
        pos.z[idx] =  new_pos_z;
    #endif //IBM_BC_Z_WALL
    #ifdef  IBM_BC_Z_PERIODIC
        pos.z[idx] = std::fmod((dfloat)(new_pos_z + NZ_TOTAL),(dfloat)(NZ_TOTAL));
    #endif //IBM_BC_Z_PERIODIC
}

__global__
void gpuForceInterpolationSpread(
    IbmNodesSoA* particlesNodes,
    ParticleCenter *pArray,
    dfloat *fMom)
{

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i >= particlesNodes->getNumNodes())
        return;

    const dfloat3SoA posNode = particlesNodes->getPos();

    ParticleCenter* pc_i = &pArray[particlesNodes->getParticleCenterIdx()[i]];

    dfloat aux, aux1; // aux variable for many things
    size_t idx; // index for many things

    const dfloat xIBM = posNode.x[i];
    const dfloat yIBM = posNode.y[i]; 
    const dfloat zIBM = posNode.z[i];

    const dfloat pos[3] = {xIBM, yIBM, zIBM};

    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];

    // First lattice position for each coordinate
    const int posBase[3] = {int(xIBM) - (P_DIST) + 1,int(yIBM) - (P_DIST) + 1, int(zIBM) - (P_DIST) + 1};

   
    // Maximum stencil index for each direction xyz ("index" to stop)
    const int maxIdx[3] = {
        #ifdef BC_X_WALL
            ((posBase[0]+P_DIST*2-1) < (int)NX)? P_DIST*2-1 : ((int)NX-1-posBase[0])
        #endif //IBM_BC_X_WALL
        #ifdef BC_X_PERIODIC
            P_DIST*2-1
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL 
            ((posBase[1]+P_DIST*2-1) < (int)NY)? P_DIST*2-1 : ((int)NY-1-posBase[1])
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
            P_DIST*2-1
        #endif //BC_Y_PERIODIC
        , 
        #ifdef BC_Z_WALL 
            ((posBase[1]+P_DIST*2-1) < (int)NZ)? P_DIST*2-1 : ((int)NZ-1-posBase[2])
        #endif //IBM_BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            P_DIST*2-1
        #endif //BC_Z_PERIODIC
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
            (posBase[2] >= 0)? 0 : -posBase[2]
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

    //compute stencil values
    for(int i = 0; i < 3; i++){
        for(int j=minIdx[i]; j <= maxIdx[i]; j++){
            stencilVal[i][j] = stencil(posBase[i]+j-(pos[i]-(i == 2? NZ*0 : 0)));
        }
    }


    dfloat rhoVar = 0;
    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    unsigned int baseIdx;
    int xx,yy,zz;

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


                #ifdef BC_X_WALL
                    xx = posBase[0] + xi;
                #endif //BC_X_WALL
                #ifdef BC_X_PERIODIC
                    xx = (posBase[0] + xi + NX)%(NX);
                #endif //BC_X_PERIODIC
                
                #ifdef BC_Y_WALL 
                    yy = posBase[1] + yj;
                #endif //BC_Y_WALL
                #ifdef BC_Y_PERIODIC    
                    yy = (posBase[1] + yj + NY)%(NY);
                #endif //IBM_BC_Y_PERIODIC
                
                #ifdef BC_Z_WALL  
                    zz = posBase[2]+zk;
                #endif //BC_Z_WALL
                #ifdef BC_Z_PERIODIC
                    zz = (posBase[2]+zk + NZ)%(NZ);
                #endif //BC_Z_PERIODIC

                baseIdx = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, 0, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);

                rhoVar += aux * (RHO_0 + fMom[baseIdx + M_RHO_INDEX]);
                uxVar  += aux * (fMom[baseIdx + BLOCK_LBM_SIZE * M_UX_INDEX]/F_M_I_SCALE);
                uyVar  += aux * (fMom[baseIdx + BLOCK_LBM_SIZE * M_UY_INDEX]/F_M_I_SCALE);
                uzVar  += aux * (fMom[baseIdx + BLOCK_LBM_SIZE * M_UZ_INDEX]/F_M_I_SCALE);
            }
        }
    }

    // Index of particle center of this particle node
    idx = particlesNodes->getParticleCenterIdx()[i];

    // Velocity on node given the particle velocity and rotation
    dfloat ux_calc = 0;
    dfloat uy_calc = 0;
    dfloat uz_calc = 0;

    // Load position of particle center
    const dfloat x_pc = pc_i->getPosX();
    const dfloat y_pc = pc_i->getPosY();
    const dfloat z_pc = pc_i->getPosZ();

    dfloat dx = xIBM - x_pc;
    dfloat dy = yIBM - y_pc;
    dfloat dz = zIBM - z_pc;

    #ifdef BC_X_PERIODIC
    if(abs(dx) > (dfloat)(NX)/2.0){
        if(dx < 0)
            dx = (xIBM + NX) - x_pc;
        else
            dx = (xIBM - NX) - x_pc;
    }
    #endif //BC_X_PERIODIC
    
    #ifdef BC_Y_PERIODIC
    if(abs(dy) > (dfloat)(NY)/2.0){
        if(dy < 0)
            dy = (yIBM + NY) - y_pc;
        else
            dy = (yIBM - NY) - y_pc;
    }
    #endif //BC_Y_PERIODIC

    #ifdef BC_Z_PERIODIC
    if(abs(dz) > (dfloat)(NZ)/2.0){
        if(dz < 0)
            dz = (zIBM + NZ) - z_pc;
        else
            dz = (zIBM - NZ) - z_pc;
    }
    #endif //BC_Z_PERIODIC

    // Calculate velocity on node if particle is movable
    if(pc_i->getMovable()){
        // Load velocity and rotation velocity of particle center
        const dfloat vx_pc = pc_i->getVelX();
        const dfloat vy_pc = pc_i->getVelY();
        const dfloat vz_pc = pc_i->getVelZ();

        const dfloat wx_pc = pc_i->getWX();
        const dfloat wy_pc = pc_i->getWY();
        const dfloat wz_pc = pc_i->getWZ();

        // velocity on node, given the center velocity and rotation
        // (i.e. no slip boundary condition velocity)
        ux_calc = vx_pc + (wy_pc * (dz) - wz_pc * (dy));
        uy_calc = vy_pc + (wz_pc * (dx) - wx_pc * (dz));
        uz_calc = vz_pc + (wx_pc * (dy) - wy_pc * (dx));
    }

    const dfloat dA = particlesNodes->getS()[i];
    aux = 2 * rhoVar * dA * IBM_THICKNESS;

    dfloat3 deltaF;
    deltaF.x = aux * (uxVar - ux_calc);
    deltaF.y = aux * (uyVar - uy_calc);
    deltaF.z = aux * (uzVar - uz_calc);

    // Calculate IBM forces
    const dfloat3SoA force = particlesNodes->getF();
    const dfloat fxIBM = force.x[i] + deltaF.x;
    const dfloat fyIBM = force.y[i] + deltaF.y;
    const dfloat fzIBM = force.z[i] + deltaF.z;

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
 
                #ifdef BC_X_WALL
                    xx = posBase[0] + xi;
                #endif //BC_X_WALL
                #ifdef BC_X_PERIODIC
                    xx = (posBase[0] + xi + NX)%(NX);
                #endif //BC_X_PERIODIC
                
                #ifdef BC_Y_WALL 
                    yy = posBase[1] + yj;
                #endif //BC_Y_WALL
                #ifdef BC_Y_PERIODIC    
                    yy = (posBase[1] + yj + NY)%(NY);
                #endif //IBM_BC_Y_PERIODIC
                
                #ifdef BC_Z_WALL  
                    zz = posBase[2]+zk;
                #endif //BC_Z_WALL
                #ifdef BC_Z_PERIODIC
                    zz = (posBase[2]+zk + NZ)%(NZ);
                #endif //BC_Z_PERIODIC

                baseIdx = idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, 0, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ);
                
                atomicAdd(&(fMom[baseIdx + M_FX_INDEX* BLOCK_LBM_SIZE]), -deltaF.x * aux);
                atomicAdd(&(fMom[baseIdx + M_FY_INDEX* BLOCK_LBM_SIZE]), -deltaF.y * aux);
                atomicAdd(&(fMom[baseIdx + M_FZ_INDEX* BLOCK_LBM_SIZE]), -deltaF.z * aux);

                //TODO: find a way to do subinterations
                //here would enter the correction of the velocity field for subiterations
                //however, on moment based, we dont have the populations to recover the original velocity
                //therefore it would directly change the velocity field and moments
                //also a problem on the lattices on the block frontier, as would be necessary to recompute the populations there

            }
        }
    }


    // Update node force
    force.x[i] = fxIBM;
    force.y[i] = fyIBM;
    force.z[i] = fzIBM;


    const dfloat3SoA delta_force = particlesNodes->getDeltaF();
    // Update node delta force
    delta_force.x[i] = deltaF.x;
    delta_force.y[i] = deltaF.y;
    delta_force.z[i] = deltaF.z;


    const dfloat3 deltaMomentum = dfloat3(
        (dy) * deltaF.z - (dz) * deltaF.y,
        (dz) * deltaF.x - (dx) * deltaF.z,
        (dx) * deltaF.y - (dy) * deltaF.x
    );
    
    atomicAdd(&(pc_i->getFXatomic()), deltaF.x);
    atomicAdd(&(pc_i->getFYatomic()), deltaF.y);
    atomicAdd(&(pc_i->getFZatomic()), deltaF.z);

    atomicAdd(&(pc_i->getMXatomic()), deltaMomentum.x);
    atomicAdd(&(pc_i->getMYatomic()), deltaMomentum.y);
    atomicAdd(&(pc_i->getMZatomic()), deltaMomentum.z);
}

// #endif //PARTICLE_MODEL