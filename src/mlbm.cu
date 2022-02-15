#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(
    dfloat *fMom, char *dNodeType,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1,
    dfloat *gGhostX_0, dfloat *gGhostX_1,
    dfloat *gGhostY_0, dfloat *gGhostY_1,
    dfloat *gGhostZ_0, dfloat *gGhostZ_1)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat pop[Q];

    // Load moments from global memory

    //rho'
    char nodeType = dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat rhoVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat ux_t30  = 3.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30  = 3.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30  = 3.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixx_t45   = 4.5*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixy_t90   = 9.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixz_t90   = 9.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyy_t45   = 4.5*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyz_t90   = 9.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pizz_t45   = 4.5*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)];

    //calculate post collision populations
    dfloat multiplyTerm;
    multiplyTerm = rhoVar * W0;
    dfloat pics2 = 1.0 - cs2 * (pixx_t45 + piyy_t45 + pizz_t45);

    pop[ 0] = multiplyTerm * (pics2);
    multiplyTerm = rhoVar * W1;
    pop[ 1] = multiplyTerm * (pics2 + ux_t30 + pixx_t45);
    pop[ 2] = multiplyTerm * (pics2 - ux_t30 + pixx_t45);
    pop[ 3] = multiplyTerm * (pics2 + uy_t30 + piyy_t45);
    pop[ 4] = multiplyTerm * (pics2 - uy_t30 + piyy_t45);
    pop[ 5] = multiplyTerm * (pics2 + uz_t30 + pizz_t45);
    pop[ 6] = multiplyTerm * (pics2 - uz_t30 + pizz_t45);
    multiplyTerm = rhoVar * W2;
    pop[ 7] = multiplyTerm * (pics2 +ux_t30 + uy_t30 + pixx_t45 + piyy_t45 + pixy_t90);
    pop[ 8] = multiplyTerm * (pics2 -ux_t30 - uy_t30 + pixx_t45 + piyy_t45 + pixy_t90);
    pop[ 9] = multiplyTerm * (pics2 +ux_t30 + uz_t30 + pixx_t45 + pizz_t45 + pixz_t90);
    pop[10] = multiplyTerm * (pics2 -ux_t30 - uz_t30 + pixx_t45 + pizz_t45 + pixz_t90);
    pop[11] = multiplyTerm * (pics2 +uy_t30 + uz_t30 + piyy_t45 + pizz_t45 + piyz_t90);
    pop[12] = multiplyTerm * (pics2 -uy_t30 - uz_t30 + piyy_t45 + pizz_t45 + piyz_t90);
    pop[13] = multiplyTerm * (pics2 +ux_t30 - uy_t30 + pixx_t45 + piyy_t45 - pixy_t90);
    pop[14] = multiplyTerm * (pics2 -ux_t30 + uy_t30 + pixx_t45 + piyy_t45 - pixy_t90);
    pop[15] = multiplyTerm * (pics2 +ux_t30 - uz_t30 + pixx_t45 + pizz_t45 - pixz_t90);
    pop[16] = multiplyTerm * (pics2 -ux_t30 + uz_t30 + pixx_t45 + pizz_t45 - pixz_t90);
    pop[17] = multiplyTerm * (pics2 +uy_t30 - uz_t30 + piyy_t45 + pizz_t45 - piyz_t90);
    pop[18] = multiplyTerm * (pics2 -uy_t30 + uz_t30 + piyy_t45 + pizz_t45 - piyz_t90);   
    #ifdef D3Q27
    multiplyTerm = rhoVar * W3;
    pop[19] = multiplyTerm * (pics2 + ux_t30 + uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 + pixz_t90 + piyz_t90));
    pop[20] = multiplyTerm * (pics2 - ux_t30 - uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 + pixz_t90 + piyz_t90));
    pop[21] = multiplyTerm * (pics2 + ux_t30 + uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 - pixz_t90 - piyz_t90));
    pop[22] = multiplyTerm * (pics2 - ux_t30 - uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 - pixz_t90 - piyz_t90));
    pop[23] = multiplyTerm * (pics2 + ux_t30 - uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 - pixz_t90 + piyz_t90));
    pop[24] = multiplyTerm * (pics2 - ux_t30 + uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 - pixz_t90 + piyz_t90));
    pop[25] = multiplyTerm * (pics2 - ux_t30 + uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 + pixz_t90 - piyz_t90));
    pop[26] = multiplyTerm * (pics2 + ux_t30 - uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 + pixz_t90 - piyz_t90));
    #endif //D3Q27

    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (Q - 1)];

    //save populations in shared memory
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = pop[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = pop[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = pop[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = pop[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = pop[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = pop[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = pop[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = pop[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = pop[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = pop[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = pop[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = pop[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = pop[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = pop[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = pop[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = pop[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = pop[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = pop[18];
    #ifdef D3Q27
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18)] = pop[19];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19)] = pop[20];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20)] = pop[21];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21)] = pop[22];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22)] = pop[23];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23)] = pop[24];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24)] = pop[25];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25)] = pop[26];
    #endif //D3Q27

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;

    pop[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    pop[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    pop[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    pop[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    pop[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    pop[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    pop[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    pop[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    pop[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    pop[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    pop[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    pop[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    pop[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    pop[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    pop[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    pop[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];
    #ifdef D3Q27
    pop[19] = s_pop[idxPopBlock(xm1, ym1, zm1, 18)];
    pop[20] = s_pop[idxPopBlock(xp1, yp1, zp1, 19)];
    pop[21] = s_pop[idxPopBlock(xm1, ym1, zp1, 20)];
    pop[22] = s_pop[idxPopBlock(xp1, yp1, zm1, 21)];
    pop[23] = s_pop[idxPopBlock(xm1, yp1, zm1, 22)];
    pop[24] = s_pop[idxPopBlock(xp1, ym1, zp1, 23)];
    pop[25] = s_pop[idxPopBlock(xp1, ym1, zm1, 24)];
    pop[26] = s_pop[idxPopBlock(xm1, yp1, zp1, 25)];
    #endif

    /* load pop from global in cover nodes */


   
    gpuInterfacePull(threadIdx,blockIdx,pop,fGhostX_0, fGhostX_1, fGhostY_0, fGhostY_1, fGhostZ_0, fGhostZ_1);

    #ifdef BC_POPULATION_BASED
        if(nodeType)
            gpuBoundaryConditionPop(threadIdx,blockIdx,pop,s_pop,nodeType); 

            //calculate streaming moments
        #ifdef D3Q19
            //equation3
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
            dfloat invRho = 1 / rhoVar;
            //equation4 + force correction
            ux_t30 = 3.0 * ((pop[ 1] + pop[7] + pop[ 9] + pop[13] + pop[15]) - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16]) + 0.5 * FX) * invRho;
            uy_t30 = 3.0 * ((pop[ 3] + pop[7] + pop[11] + pop[14] + pop[17]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18]) + 0.5 * FY) * invRho;
            uz_t30 = 3.0 * ((pop[ 5] + pop[9] + pop[11] + pop[16] + pop[18]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17]) + 0.5 * FZ) * invRho;

            //equation5
            pixx_t45 = 4.5 * ( (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2);
            pixy_t90 = 9.0 * (((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho);
            pixz_t90 = 9.0 * (((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho);
            piyy_t45 = 4.5 * ( (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2);
            piyz_t90 = 9.0 * (((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho);
            pizz_t45 = 4.5 * ( (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2);


        #endif
        #ifdef D3Q27
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
            dfloat invRho = 1 / rhoVar;
            ux_t30 = 3.0 * ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho*3.0;
            uy_t30 = 3.0 * ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho*3.0;
            uz_t30 = 3.0 * ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho*3.0;

            pixx_t45 = 4.5 * ( (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho- cs2);
            pixy_t90 = 9.0 * (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) * invRho);
            pixz_t90 = 9.0 * (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) * invRho);
            piyy_t45 = 4.5 * ( (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho- cs2);
            piyz_t90 = 9.0 * (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24]))*invRho);
            pizz_t45 = 4.5 * ( (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]) * invRho- cs2);
        #endif
    #endif

    #ifdef BC_MOMENT_BASED
        dfloat invRho;
        if(nodeType){
            gpuBoundaryConditionMom(pop,rhoVar,nodeType,ux_t30,uy_t30,uz_t30,pixx_t45,pixy_90,pixz_90,piyy_t45,piyz_90,pizz_t45);
            invRho = 1.0 / rhoVar;
        }else{

            //calculate streaming moments
            #ifdef D3Q19
                //equation3
                rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
                invRho = 1 / rhoVar;
                //equation4 + force correction
                ux_t30 = 3.0 * ((pop[ 1] + pop[7] + pop[ 9] + pop[13] + pop[15]) - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16]) + 0.5 * FX) * invRho;
                uy_t30 = 3.0 * ((pop[ 3] + pop[7] + pop[11] + pop[14] + pop[17]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18]) + 0.5 * FY) * invRho;
                uz_t30 = 3.0 * ((pop[ 5] + pop[9] + pop[11] + pop[16] + pop[18]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17]) + 0.5 * FZ) * invRho;

                //equation5
                pixx_t45 = 4.5 * ( (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2);
                pixy_t90 = 9.0 * (((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho);
                pixz_t90 = 9.0 * (((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho);
                piyy_t45 = 4.5 * ( (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2);
                piyz_t90 = 9.0 * (((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho);
                pizz_t45 = 4.5 * ( (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2);


            #endif
            #ifdef D3Q27
                rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
                dfloat invRho = 1 / rhoVar;
                ux_t30 = 3.0 * ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
                uy_t30 = 3.0 * ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
                uz_t30 = 3.0 * ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;

                pixx_t45 = 4.5 * ( (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2) * invRho);
                pixy_t90 = 9.0 * (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) * invRho);
                pixz_t90 = 9.0 * (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) * invRho);
                piyy_t45 = 4.5 * ( (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2) * invRho);
                piyz_t90 = 9.0 * (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24]))*invRho);
                pizz_t45 = 4.5 * ( (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2) * invRho);
            #endif
        }
    #endif // moment based

    //NOTE : STREAMING DONE, NOW COLLIDE

    //Collide Moments
    //Equiblibrium momements
    dfloat invRho_mt15 = -1.5*invRho;
    ux_t30 = (T_OMEGA * (ux_t30 + invRho_mt15 * FX ) + OMEGA * ux_t30 + TT_OMEGA_T3 * FX);
    uy_t30 = (T_OMEGA * (uy_t30 + invRho_mt15 * FY ) + OMEGA * uy_t30 + TT_OMEGA_T3 * FY);
    uz_t30 = (T_OMEGA * (uz_t30 + invRho_mt15 * FZ ) + OMEGA * uz_t30 + TT_OMEGA_T3 * FZ);
    
    //equation 90
    pixx_t45 = (T_OMEGA * pixx_t45  +   OMEGAd2 * ux_t30 * ux_t30    - invRho_mt15 * TT_OMEGA * (FX * ux_t30 + FX * ux_t30));
    piyy_t45 = (T_OMEGA * piyy_t45  +   OMEGAd2 * uy_t30 * uy_t30    - invRho_mt15 * TT_OMEGA * (FY * uy_t30 + FY * uy_t30));
    pizz_t45 = (T_OMEGA * pizz_t45  +   OMEGAd2 * uz_t30 * uz_t30    - invRho_mt15 * TT_OMEGA * (FZ * uz_t30 + FZ * uz_t30));

    pixy_t90 = (T_OMEGA * pixy_t90  +     OMEGA * ux_t30 * uy_t30    +    TT_OMEGA_T3 *invRho* (FX * uy_t30 + FY * ux_t30));
    pixz_t90 = (T_OMEGA * pixz_t90  +     OMEGA * ux_t30 * uz_t30    +    TT_OMEGA_T3 *invRho* (FX * uz_t30 + FZ * ux_t30));
    piyz_t90 = (T_OMEGA * piyz_t90  +     OMEGA * uy_t30 * uz_t30    +    TT_OMEGA_T3 *invRho* (FY * uz_t30 + FZ * uy_t30));

    //calculate post collision populations
    multiplyTerm = rhoVar * W0;
    pics2 = 1.0 - cs2 * (pixx_t45 + piyy_t45 + pizz_t45);

    pop[ 0] = multiplyTerm * (pics2);
    multiplyTerm = rhoVar * W1;
    pop[ 1] = multiplyTerm * (pics2 + ux_t30 + pixx_t45);
    pop[ 2] = multiplyTerm * (pics2 - ux_t30 + pixx_t45);
    pop[ 3] = multiplyTerm * (pics2 + uy_t30 + piyy_t45);
    pop[ 4] = multiplyTerm * (pics2 - uy_t30 + piyy_t45);
    pop[ 5] = multiplyTerm * (pics2 + uz_t30 + pizz_t45);
    pop[ 6] = multiplyTerm * (pics2 - uz_t30 + pizz_t45);
    multiplyTerm = rhoVar * W2;
    pop[ 7] = multiplyTerm * (pics2 + ( ux_t30 + uy_t30) + (pixx_t45 + piyy_t45) + pixy_t90);
    pop[ 8] = multiplyTerm * (pics2 + (-ux_t30 - uy_t30) + (pixx_t45 + piyy_t45) + pixy_t90);
    pop[ 9] = multiplyTerm * (pics2 + ( ux_t30 + uz_t30) + (pixx_t45 + pizz_t45) + pixz_t90);
    pop[10] = multiplyTerm * (pics2 + (-ux_t30 - uz_t30) + (pixx_t45 + pizz_t45) + pixz_t90);
    pop[11] = multiplyTerm * (pics2 + ( uy_t30 + uz_t30) + (piyy_t45 + pizz_t45) + piyz_t90);
    pop[12] = multiplyTerm * (pics2 + (-uy_t30 - uz_t30) + (piyy_t45 + pizz_t45) + piyz_t90);
    pop[13] = multiplyTerm * (pics2 + ( ux_t30 - uy_t30) + (pixx_t45 + piyy_t45) - pixy_t90);
    pop[14] = multiplyTerm * (pics2 + (-ux_t30 + uy_t30) + (pixx_t45 + piyy_t45) - pixy_t90);
    pop[15] = multiplyTerm * (pics2 + ( ux_t30 - uz_t30) + (pixx_t45 + pizz_t45) - pixz_t90);
    pop[16] = multiplyTerm * (pics2 + (-ux_t30 + uz_t30) + (pixx_t45 + pizz_t45) - pixz_t90);
    pop[17] = multiplyTerm * (pics2 + ( uy_t30 - uz_t30) + (piyy_t45 + pizz_t45) - piyz_t90);
    pop[18] = multiplyTerm * (pics2 + (-uy_t30 + uz_t30) + (piyy_t45 + pizz_t45) - piyz_t90);   
    #ifdef D3Q27
    multiplyTerm = rhoVar * W3;
    pop[19] = multiplyTerm * (pics2 + ux_t30 + uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 + pixz_t90 + piyz_t90));
    pop[20] = multiplyTerm * (pics2 - ux_t30 - uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 + pixz_t90 + piyz_t90));
    pop[21] = multiplyTerm * (pics2 + ux_t30 + uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 - pixz_t90 - piyz_t90));
    pop[22] = multiplyTerm * (pics2 - ux_t30 - uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 + (pixy_t90 - pixz_t90 - piyz_t90));
    pop[23] = multiplyTerm * (pics2 + ux_t30 - uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 - pixz_t90 + piyz_t90));
    pop[24] = multiplyTerm * (pics2 - ux_t30 + uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 - pixz_t90 + piyz_t90));
    pop[25] = multiplyTerm * (pics2 - ux_t30 + uyVar_t30 + uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 + pixz_t90 - piyz_t90));
    pop[26] = multiplyTerm * (pics2 + ux_t30 - uyVar_t30 - uzVar_t30 + pixx_t45 + piyy_t45 + pizz_t45 - (pixy_t90 + pixz_t90 - piyz_t90));
    #endif //D3Q27
    
    /* write to global mom */

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30/3.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30/3.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30/3.0;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)] = pixx_t45/4.5;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)] = pixy_t90/9.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)] = pixz_t90/9.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)] = piyy_t45/4.5;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)] = piyz_t90/9.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)] = pizz_t45/4.5;

    /* write to global pop */
    gpuInterfacePush(threadIdx, blockIdx, pop, gGhostX_0, gGhostX_1, gGhostY_0, gGhostY_1, gGhostZ_0, gGhostZ_1);

}
