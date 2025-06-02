#include "tracer.cuh"

__host__
void updateParticlePos(
    dfloat3 *d_particlePos, 
    dfloat3 *h_particlePos, 
    dfloat *fMom,
    cudaStream_t streamParticles,
    unsigned int step
){

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    
    const unsigned int threadsNodes = 64;
    const unsigned int gridNodes = NUM_PARTICLES % threadsNodes ? NUM_PARTICLES / threadsNodes + 1 : NUM_PARTICLES / threadsNodes;

    checkCudaErrors(cudaStreamSynchronize(streamParticles));
    velocityInterpolation<<<gridNodes, threadsNodes, 0, streamParticles>>>(d_particlePos, fMom,step);
    checkCudaErrors(cudaStreamSynchronize(streamParticles));

    bool PARTICLE_TRACER_SAVE = false; //quick fix for now
#pragma warning(push)
#pragma warning(disable: 4804)
    if(!(step%PARTICLE_TRACER_SAVE)){
        checkCudaErrors(cudaMemcpy(h_particlePos, d_particlePos, sizeof(dfloat3)*NUM_PARTICLES, cudaMemcpyDeviceToHost)); 
        saveParticleInfo(h_particlePos,step);
    }
#pragma warning(pop)
}

__global__
void velocityInterpolation(
    dfloat3 *d_particlePos, 
    dfloat *fMom,
    unsigned int step
){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= NUM_PARTICLES)
        return;

    dfloat aux, aux1;
    
    // Calculate stencils to use and the valid interval [xyz][idx]
    dfloat stencilVal[3][P_DIST*2];

    dfloat xPos = d_particlePos[i].x;
    dfloat yPos = d_particlePos[i].y;
    dfloat zPos = d_particlePos[i].z;
    dfloat pos[3] = {xPos, yPos, zPos};

    const int posBase[3] = { 
        int(xPos) - (P_DIST) + 1, 
        int(yPos) - (P_DIST) + 1, 
        int(zPos) - (P_DIST) + 1
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


    //Update particle position
    d_particlePos[i].x += uxVar;
    d_particlePos[i].y += uyVar;
    d_particlePos[i].z += uzVar;

    //AVOID THAT THE PARTICLES GO OUTSIDE OF THE DOMAIN
    #ifdef BC_X_WALL
    if(d_particlePos[i].x < 0){
        d_particlePos[i].x = 0.01;
    }
    if(d_particlePos[i].x > NX-1){
        d_particlePos[i].x = NX-1.01;
    }
    #endif

    #ifdef BC_Y_WALL
    if(d_particlePos[i].y < 0){
        d_particlePos[i].y = 0.01;
    }
    if(d_particlePos[i].y > NY-1){
        d_particlePos[i].y = NY-1.01;
    }
    #endif

    #ifdef BC_Z_WALL
    if(d_particlePos[i].z < 0){
        d_particlePos[i].z = 0.01;
    }
    if(d_particlePos[i].z > NZ-1){
        d_particlePos[i].z = NZ-1.01;
    }
    #endif

}

__host__
void saveParticleInfo(dfloat3 *h_particlePos, unsigned int step){
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