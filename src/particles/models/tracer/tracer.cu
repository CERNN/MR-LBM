#ifdef PARTICLE_MODEL

#include "tracer.cuh"

__host__ 
void tracerSimulation(
    ParticlesSoA *particles,
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
){

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    MethodRange range = particles->getMethodRange(TRACER);

    int numTracerParticles = range.last - range.first + 1;
    const unsigned int threadsNodes = 64;
    const unsigned int gridNodes = (numTracerParticles + threadsNodes - 1) / threadsNodes;

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

    tracer_positionUpdate<<<gridNodes, threadsNodes, 0, streamParticles>>>(
        pArray, fMom,range.first,range.last,step
    );
    
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        printf("Error in kernel tracer_positionUpdate: %s\n", cudaGetErrorString(kernelError));
    }

    checkCudaErrors(cudaStreamSynchronize(streamParticles));

    #ifdef PARTICLE_TRACER_SAVE
#pragma warning(push)
#pragma warning(disable: 4804)
    if(!(step%PARTICLE_TRACER_SAVE)){
        checkCudaErrors(cudaMemcpy(h_particlePos, d_particlePos, sizeof(dfloat3)*NUM_PARTICLES, cudaMemcpyDeviceToHost)); 
        saveParticleInfo(h_particlePos,step);
    }
#pragma warning(pop)
    #endif


}

__global__
void tracer_positionUpdate(
    ParticleCenter *pArray,
    dfloat *fMom,
    int firstIndex,
    int lastIndex,
    unsigned int step
){
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
    
    dfloat aux, aux1;
    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];

    ParticleCenter* pc_i = &pArray[globalIdx];

    dfloat3 pc_pos = pc_i->getPos();

    dfloat pos[3] = {pc_pos.x,pc_pos.y,pc_pos.z};
    
    const int posBase[3] = { 
        int(pos[0]) - (P_DIST) + 1, 
        int(pos[1]) - (P_DIST) + 1, 
        int(pos[2]) - (P_DIST) + 1
    };

    const int maxIdx[3] = {
        #ifdef BC_X_WALL
            ((posBase[0]+P_DIST*2-1) < (int)NX)? P_DIST*2-1 : ((int)NX-1-posBase[0])
        #endif //BC_X_WALL
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
            ((posBase[2]+P_DIST*2-1) < (int)NZ)? P_DIST*2-1 : ((int)NZ-1-posBase[2])
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            P_DIST*2-1
        #endif //BC_Z_PERIODIC
    };

    const int minIdx[3] = {
        #ifdef BC_X_WALL
            (posBase[0] >= 0)? 0 : -posBase[0]
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC
            0
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL 
            (posBase[1] >= 0)? 0 : -posBase[1]
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
            0
        #endif //BC_Y_PERIODIC
        , 
        #ifdef BC_Z_WALL 
            (posBase[2] >= 0)? 0 : -posBase[2]
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            0
        #endif //BC_Z_PERIODIC
    };

     // Particle stencil out of the domain
    if(maxIdx[0] <= 0 || maxIdx[1] <= 0 || maxIdx[2] <= 0)
        return;
    // Particle stencil out of the domain
    if(minIdx[0] >= P_DIST*2 || minIdx[1] >= P_DIST*2 || minIdx[2] >= P_DIST*2)
        return;

    //calculate velocity interpolation stencil
    for(int i = 0; i < 3; i++){
        for(int j=minIdx[i]; j <= maxIdx[i]; j++){
            stencilVal[i][j] = stencil(posBase[i]+j-(pos[i]));
        }
    }

    dfloat uxVar = 0;
    dfloat uyVar = 0;
    dfloat uzVar = 0;

    int posX, posY,posZ;
    // Interpolation 
    
    for (int zk = minIdx[2]; zk <= maxIdx[2]; zk++) // z
    {
        for (int yj = minIdx[1]; yj <= maxIdx[1]; yj++) // y
        {
            aux1 = stencilVal[2][zk]*stencilVal[1][yj];
            for (int xi = minIdx[0]; xi <= maxIdx[0]; xi++) // x
            {
                // Dirac delta (kernel)
                aux = aux1 * stencilVal[0][xi];

                //printf("%d %e \n",i,aux1);

                posX =                     
                    #ifdef BC_X_WALL
                    posBase[0]+xi
                    #endif //IBM_BC_X_WALL
                    #ifdef BC_X_PERIODIC
                        (posBase[0]+xi + NX)%(NX)
                    #endif //BC_X_PERIODIC
                    ;
                posY = 
                    #ifdef BC_Y_WALL 
                        posBase[1]+yj
                    #endif //BC_Y_WALL
                    #ifdef BC_Y_PERIODIC    
                        (posBase[1]+yj + NY)%(NY)
                    #endif //BC_Y_PERIODIC
                    ;
                posZ = 
                    #ifdef BC_Z_WALL  
                        posBase[2]+zk
                    #endif //BC_Z_WALL
                    #ifdef BC_Z_PERIODIC
                        (posBase[2]+zk + NZ)%(NZ)
                    #endif //BC_Z_PERIODIC
                    ;

    
                uxVar += fMom[idxMom(posX%BLOCK_NX, posY%BLOCK_NY, posZ%BLOCK_NZ, M_UX_INDEX, posX/BLOCK_NX, posY/BLOCK_NY, posZ/BLOCK_NZ)] * aux;
                uyVar += fMom[idxMom(posX%BLOCK_NX, posY%BLOCK_NY, posZ%BLOCK_NZ, M_UY_INDEX, posX/BLOCK_NX, posY/BLOCK_NY, posZ/BLOCK_NZ)] * aux;
                uzVar += fMom[idxMom(posX%BLOCK_NX, posY%BLOCK_NY, posZ%BLOCK_NZ, M_UZ_INDEX, posX/BLOCK_NX, posY/BLOCK_NY, posZ/BLOCK_NZ)] * aux;
            }
        }
    }

    dfloat3 dVel = dfloat3(uxVar,uyVar,uzVar);// 0.01
    //Update particle position
    pc_i->setPos(pc_i->getPos() + dVel);

      //AVOID THAT THE PARTICLES GO OUTSIDE OF THE DOMAIN
    #ifdef BC_X_WALL
    if (pc_i->getPosX() < 0) {
        pc_i->setPosX(0.01);
    }
    if (pc_i->getPosX() > NX - 1) {
        pc_i->setPosX(NX - 1.01);
    }
    #endif

    #ifdef BC_Y_WALL
    if (pc_i->getPosY() < 0) {
        pc_i->setPosY(0.01);
    }
    if (pc_i->getPosY() > NY - 1) {
        pc_i->setPosY(NY - 1.01);
    }
    #endif

    #ifdef BC_Z_WALL
    if (pc_i->getPosZ() < 0) {
        pc_i->setPosZ(0.01);
    }
    if (pc_i->getPosZ() > NZ - 1) {
        pc_i->setPosZ(NZ - 1.01);
    }
    #endif 

}

__host__
void tracer_saveParticleInfo(dfloat3 *h_particlePos, unsigned int step){
    // Names of file to save particle info
    std::string strFileParticleData = getVarFilename("particleData", step, ".csv");

    // File to save particle info
    std::ofstream outFileParticleData(strFileParticleData.c_str());

    // String with all values as csv
    std::ostringstream strValuesParticles("");
    strValuesParticles << std::scientific;

    // csv separator
    std::string sep = ",";

    std::string strColumnNames = "pIndex" + sep + "step" + sep;
    strColumnNames += "pos_x" + sep  + "pos_y" + sep  + "pos_z";
    strColumnNames += "\n";

    for(int p = 0; p < NUM_PARTICLES; p++){
        strValuesParticles << p << sep;
        strValuesParticles << step << sep;
        strValuesParticles << h_particlePos[p].x << sep << h_particlePos[p].y << sep << h_particlePos[p].z;
        strValuesParticles << "\n";
    }

    outFileParticleData << strColumnNames << strValuesParticles.str();
}

#endif //PARTICLE_MODEL