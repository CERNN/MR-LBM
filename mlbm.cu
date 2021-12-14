#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(
    dfloat *fMom,
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
    dfloat rhoVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
    //u'
    dfloat uxVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uyVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uzVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
    //m'
    dfloat pixx   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixy   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixz   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyy   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyz   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pizz   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)];

    //calculate post collision populations
    #pragma unroll //equation 10
    for (int i = 0; i < Q; i++)
    {
        pop[i] = rhoVar * w[i] * (1 
        + as2 * (uxVar * cx[i] + uyVar * cy[i] + uzVar * cz[i]) 
        + 0.5 * as2 * as2 * (
            pixx * (cx[i] * cx[i] - cs2) + 
            2.0*pixy * (cx[i] * cy[i]) + 
            2.0*pixz * (cx[i] * cz[i]) + 
            piyy * (cy[i] * cy[i] - cs2) + 
            2.0*piyz * (cy[i] * cz[i]) + 
            pizz * (cz[i] * cz[i] - cs2))
        );
    }

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

    gpuInterfacePullOffset(threadIdx,blockIdx,pop,fGhostX_0, fGhostX_1, fGhostY_0, fGhostY_1, fGhostZ_0, fGhostZ_1);


//calculate streaming moments
    #ifdef D3Q19
        //equation3
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        dfloat invRho = 1 / rhoVar;
        //equation4 + force correction
        uxVar = ((pop[ 1] + pop[7] + pop[ 9] + pop[13] + pop[15]) - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16]) + 0.5 * FX) * invRho;
        uyVar = ((pop[ 3] + pop[7] + pop[11] + pop[14] + pop[17]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18]) + 0.5 * FY) * invRho;
        uzVar = ((pop[ 5] + pop[9] + pop[11] + pop[16] + pop[18]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17]) + 0.5 * FZ) * invRho;

        //equation5
        pixx =  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] -  cs2) * invRho;
        pixy = ((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho;
        pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
        piyy =  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] - cs2) * invRho;
        piyz = ((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho;
        pizz =  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] - cs2) * invRho;


    #endif
    #ifdef D3Q27
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        dfloat invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;

        pixx =  (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2) * invRho;
        pixy = ((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) * invRho;
        pixz = ((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) * invRho;
        piyy =  (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2) * invRho;
        piyz = ((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24]))*invRho;
        pizz =  (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26] - cs2) * invRho;
    #endif

    //NOTE : STREAMING DONE, NOW COLLIDE

    //Collide Moments
    //Equiblibrium momements
    
    //equation 90
    pixx = (T_OMEGA * rhoVar * (pixx) + OMEGA * rhoVar * (uxVar * uxVar) + TT_OMEGA * (FX * uxVar + FX * uxVar))*invRho;
    pixy = (T_OMEGA * rhoVar * (pixy) + OMEGA * rhoVar * (uxVar * uyVar) + TT_OMEGA * (FX * uyVar + FY * uxVar))*invRho;
    pixz = (T_OMEGA * rhoVar * (pixz) + OMEGA * rhoVar * (uxVar * uzVar) + TT_OMEGA * (FX * uzVar + FZ * uxVar))*invRho;
    piyy = (T_OMEGA * rhoVar * (piyy) + OMEGA * rhoVar * (uyVar * uyVar) + TT_OMEGA * (FY * uyVar + FY * uyVar))*invRho;
    piyz = (T_OMEGA * rhoVar * (piyz) + OMEGA * rhoVar * (uyVar * uzVar) + TT_OMEGA * (FY * uzVar + FZ * uyVar))*invRho;
    pizz = (T_OMEGA * rhoVar * (pizz) + OMEGA * rhoVar * (uzVar * uzVar) + TT_OMEGA * (FZ * uzVar + FZ * uzVar))*invRho;

    for (int i = 0; i < Q; i++)
    {
        pop[i] = rhoVar * w[i] * (1 
        + as2 * (uxVar * cx[i] + uyVar * cy[i] + uzVar * cz[i]) 
        + 0.5 * as2 * as2 * (
            pixx * (cx[i] * cx[i] - cs2) + 
            2.0*pixy * (cx[i] * cy[i]) + 
            2.0*pixz * (cx[i] * cz[i]) + 
            piyy * (cy[i] * cy[i] - cs2) + 
            2.0*piyz * (cy[i] * cz[i]) + 
            pizz * (cz[i] * cz[i] - cs2))
        );
    }

    /* write to global mom */

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)] = uxVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)] = uyVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)] = uzVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)] = pixx;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)] = pixy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)] = pixz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)] = piyy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)] = piyz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)] = pizz;

    /* write to global pop */
    gpuInterfacePushCentered(threadIdx, blockIdx, pop, gGhostX_0, gGhostX_1, gGhostY_0, gGhostY_1, gGhostZ_0, gGhostZ_1);

}
