#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(
    dfloat *fMom,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat pop[Q];

    // Load moments from global memory

    dfloat rhoVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uxVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uyVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uzVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixx   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixy   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixz   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyy   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyz   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pizz   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat fNodeEq[Q];
    dfloat fNodeNeq[Q];

    // Moments

    dfloat pixx_eq = rhoVar * (uxVar * uxVar + cs2);
    dfloat pixy_eq = rhoVar * (uxVar * uyVar);
    dfloat pixz_eq = rhoVar * (uxVar * uzVar);
    dfloat piyy_eq = rhoVar * (uyVar * uyVar + cs2);
    dfloat piyz_eq = rhoVar * (uyVar * uzVar);
    dfloat pizz_eq = rhoVar * (uzVar * uzVar + cs2);

    // Calculate temporary variables
    dfloat p1_muu15 = 1 - 1.5 * (uxVar * uxVar + uyVar * uyVar + uzVar * uzVar);
    dfloat rhoW0 = rhoVar * W0;
    dfloat rhoW1 = rhoVar * W1;
    dfloat rhoW2 = rhoVar * W2;
    dfloat W1t3d2 = W1 * 3.0 / 2.0;
    dfloat W2t3d2 = W2 * 3.0 / 2.0;
    dfloat W1t9d2 = W1t3d2 * 3.0;
    dfloat W2t9d2 = W2t3d2 * 3.0;

    #ifdef D3Q27
    dfloat rhoW3 = rhoVar * W3;
    dfloat W3t9d2 = W3 * 9 / 2;
    #endif
    dfloat ux3 = 3 * uxVar;
    dfloat uy3 = 3 * uyVar;
    dfloat uz3 = 3 * uzVar;

    // Calculate equilibrium fNodeEq
    fNodeEq[ 0] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNodeEq[ 1] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNodeEq[ 2] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNodeEq[ 3] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNodeEq[ 4] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNodeEq[ 5] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNodeEq[ 6] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNodeEq[ 7] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNodeEq[ 8] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNodeEq[ 9] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
    fNodeEq[10] = gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15);
    fNodeEq[11] = gpu_f_eq(rhoW2, uy3 + uz3, p1_muu15);
    fNodeEq[12] = gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15);
    fNodeEq[13] = gpu_f_eq(rhoW2, ux3 - uy3, p1_muu15);
    fNodeEq[14] = gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15);
    fNodeEq[15] = gpu_f_eq(rhoW2, ux3 - uz3, p1_muu15);
    fNodeEq[16] = gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15);
    fNodeEq[17] = gpu_f_eq(rhoW2, uy3 - uz3, p1_muu15);
    fNodeEq[18] = gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15);
#ifdef D3Q27
    fNodeEq[19] = gpu_f_eq(rhoW3, ux3 + uy3 + uz3, p1_muu15);
    fNodeEq[20] = gpu_f_eq(rhoW3, -ux3 - uy3 - uz3, p1_muu15);
    fNodeEq[21] = gpu_f_eq(rhoW3, ux3 + uy3 - uz3, p1_muu15);
    fNodeEq[22] = gpu_f_eq(rhoW3, -ux3 - uy3 + uz3, p1_muu15);
    fNodeEq[23] = gpu_f_eq(rhoW3, ux3 - uy3 + uz3, p1_muu15);
    fNodeEq[24] = gpu_f_eq(rhoW3, -ux3 + uy3 - uz3, p1_muu15);
    fNodeEq[25] = gpu_f_eq(rhoW3, -ux3 + uy3 + uz3, p1_muu15);
    fNodeEq[26] = gpu_f_eq(rhoW3, ux3 - uy3 - uz3, p1_muu15);
#endif

// CALCULATE NON-EQUILIBRIUM POPULATIONS
#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        fNodeNeq[i] = rhoVar * 1.5 * w[i] *
                          (((cx[i] * cx[i] - cs2) * (pixx - pixx_eq) + //Q-iab*(m_ab - m_ab^eq)
                        2 * (cx[i] * cy[i])       * (pixy - pixy_eq) +
                        2 * (cx[i] * cz[i])       * (pixz - pixz_eq) +
                            (cy[i] * cy[i] - cs2) * (piyy - piyy_eq) +
                        2 * (cy[i] * cz[i])       * (piyz - piyz_eq) +
                            (cz[i] * cz[i] - cs2) * (pizz - pizz_eq)) -
                       cs2 * (cx[i] * FX + cy[i] * FY + cz[i] * FZ)); //force term
    }

    //CALCULATE COLLISION POPULATIONS
    pop[ 0] = fNodeEq[ 0] + fNodeNeq[ 0];
    pop[ 1] = fNodeEq[ 1] + fNodeNeq[ 1];
    pop[ 2] = fNodeEq[ 2] + fNodeNeq[ 2];
    pop[ 3] = fNodeEq[ 3] + fNodeNeq[ 3];
    pop[ 4] = fNodeEq[ 4] + fNodeNeq[ 4];
    pop[ 5] = fNodeEq[ 5] + fNodeNeq[ 5];
    pop[ 6] = fNodeEq[ 6] + fNodeNeq[ 6];
    pop[ 7] = fNodeEq[ 7] + fNodeNeq[ 7];
    pop[ 8] = fNodeEq[ 8] + fNodeNeq[ 8];
    pop[ 9] = fNodeEq[ 9] + fNodeNeq[ 9];
    pop[10] = fNodeEq[10] + fNodeNeq[10];
    pop[11] = fNodeEq[11] + fNodeNeq[11];
    pop[12] = fNodeEq[12] + fNodeNeq[12];
    pop[13] = fNodeEq[13] + fNodeNeq[13];
    pop[14] = fNodeEq[14] + fNodeNeq[14];
    pop[15] = fNodeEq[15] + fNodeNeq[15];
    pop[16] = fNodeEq[16] + fNodeNeq[16];
    pop[17] = fNodeEq[17] + fNodeNeq[17];
    pop[18] = fNodeEq[18] + fNodeNeq[18];

#ifdef D3Q27
    pop[19] = fNodeEq[19] + fNodeNeq[19];
    pop[20] = fNodeEq[20] + fNodeNeq[20];
    pop[21] = fNodeEq[21] + fNodeNeq[21];
    pop[22] = fNodeEq[22] + fNodeNeq[22];
    pop[23] = fNodeEq[23] + fNodeNeq[23];
    pop[24] = fNodeEq[24] + fNodeNeq[24];
    pop[25] = fNodeEq[25] + fNodeNeq[25];
    pop[26] = fNodeEq[26] + fNodeNeq[26];
#endif

    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (Q-1)];

    //save populations in shared memory
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 0)] = pop[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1)] = pop[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2)] = pop[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3)] = pop[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4)] = pop[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5)] = pop[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6)] = pop[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7)] = pop[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8)] = pop[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9)] = pop[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,10)] = pop[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,11)] = pop[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,12)] = pop[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,13)] = pop[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,14)] = pop[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,15)] = pop[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,16)] = pop[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,17)] = pop[18];
    #ifdef D3Q27
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,18)] = pop[19];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,19)] = pop[20];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,20)] = pop[21];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,21)] = pop[22];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,22)] = pop[23];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,23)] = pop[24];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,24)] = pop[25];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,25)] = pop[26];
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

    if (threadIdx.x == 0)
    {
        pop[ 1] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 7] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 9] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[13] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[15] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[19] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[21] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[23] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[26] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    else if (threadIdx.x == BLOCK_NX - 1)
    {
        pop[ 2] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 8] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[10] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[14] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[16] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[20] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[22] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[24] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[25] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    if (threadIdx.y == 0)
    {
        pop[ 3] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 7] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[11] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[14] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[17] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[19] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[21] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[24] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[25] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    else if (threadIdx.y == BLOCK_NY - 1)
    {
        pop[ 4] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 8] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[12] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[13] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[18] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[20] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[22] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[23] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[26] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    if (threadIdx.z == 0)
    {
        pop[ 5] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 9] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[11] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[16] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[18] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[19] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[22] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[23] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[25] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    else if (threadIdx.z == BLOCK_NZ - 1)
    {
        pop[ 6] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[10] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[12] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[15] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[17] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[20] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[21] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[24] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[26] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }

    //if( x == 0 && y==0 && z ==0)
    //    printf("\n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f \n%f\n%f \n", 
    //    pop[0] , pop[1] , pop[2] , pop[3] , pop[4] , pop[5] , pop[6] , 
    //    pop[7] , pop[8] , pop[9] , pop[10] , pop[11] , pop[12] , pop[13] , pop[14] , pop[15] , pop[16] , pop[17] , pop[18]);

    #ifdef D3Q19
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        dfloat invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16]) + 0.5 * FX) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18]) + 0.5 * FY) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17]) + 0.5 * FZ) * invRho;
    #endif
    #ifdef D3Q27
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        dfloat invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;
    #endif


    //NOTE : STREAMING DONE, NOW COLLIDE

    //Collide Moments
    //Equiblibrium momements
    dfloat mNodeEq[6];
    mNodeEq[0] = rhoVar * (uxVar * uxVar + cs2);
    mNodeEq[1] = rhoVar * (uxVar * uyVar);
    mNodeEq[2] = rhoVar * (uxVar * uzVar);
    mNodeEq[3] = rhoVar * (uyVar * uyVar + cs2);
    mNodeEq[4] = rhoVar * (uyVar * uzVar);
    mNodeEq[5] = rhoVar * (uzVar * uzVar + cs2);

    pixx = pixx - OMEGA * (pixx - mNodeEq[0]) + TT_OMEGA * (FX * uxVar + FX * uxVar);
    pixy = pixy - OMEGA * (pixy - mNodeEq[1]) + TT_OMEGA * (FX * uyVar + FY * uxVar);
    pixz = pixz - OMEGA * (pixz - mNodeEq[2]) + TT_OMEGA * (FX * uzVar + FZ * uxVar);
    piyy = piyy - OMEGA * (piyy - mNodeEq[3]) + TT_OMEGA * (FY * uyVar + FY * uyVar);
    piyz = piyz - OMEGA * (piyz - mNodeEq[4]) + TT_OMEGA * (FY * uzVar + FZ * uyVar);
    pizz = pizz - OMEGA * (pizz - mNodeEq[5]) + TT_OMEGA * (FZ * uzVar + FZ * uzVar);

    pixx_eq = rhoVar * (uxVar * uxVar + cs2);
    pixy_eq = rhoVar * (uxVar * uyVar);
    pixz_eq = rhoVar * (uxVar * uzVar);
    piyy_eq = rhoVar * (uyVar * uyVar + cs2);
    piyz_eq = rhoVar * (uyVar * uzVar);
    pizz_eq = rhoVar * (uzVar * uzVar + cs2);

    // Calculate temporary variables
    p1_muu15 = 1 - 1.5 * (uxVar * uxVar + uyVar * uyVar + uzVar * uzVar);
    rhoW0 = rhoVar * W0;
    rhoW1 = rhoVar * W1;
    rhoW2 = rhoVar * W2;
    W1t3d2 = W1 * 3.0 / 2.0;
    W2t3d2 = W2 * 3.0 / 2.0;
    W1t9d2 = W1t3d2 * 3.0;
    W2t9d2 = W2t3d2 * 3.0;

#ifdef D3Q27
    rhoW3 = rhoVar * W3;
    W3t9d2 = W3 * 9 / 2;
#endif
    ux3 = 3 * uxVar;
    uy3 = 3 * uyVar;
    uz3 = 3 * uzVar;

    // Calculate equilibrium fNodeEq
    fNodeEq[ 0] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNodeEq[ 1] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNodeEq[ 2] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNodeEq[ 3] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNodeEq[ 4] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNodeEq[ 5] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNodeEq[ 6] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNodeEq[ 7] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNodeEq[ 8] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNodeEq[ 9] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
    fNodeEq[10] = gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15);
    fNodeEq[11] = gpu_f_eq(rhoW2, uy3 + uz3, p1_muu15);
    fNodeEq[12] = gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15);
    fNodeEq[13] = gpu_f_eq(rhoW2, ux3 - uy3, p1_muu15);
    fNodeEq[14] = gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15);
    fNodeEq[15] = gpu_f_eq(rhoW2, ux3 - uz3, p1_muu15);
    fNodeEq[16] = gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15);
    fNodeEq[17] = gpu_f_eq(rhoW2, uy3 - uz3, p1_muu15);
    fNodeEq[18] = gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15);
#ifdef D3Q27
    fNodeEq[19] = gpu_f_eq(rhoW3, ux3 + uy3 + uz3, p1_muu15);
    fNodeEq[20] = gpu_f_eq(rhoW3, -ux3 - uy3 - uz3, p1_muu15);
    fNodeEq[21] = gpu_f_eq(rhoW3, ux3 + uy3 - uz3, p1_muu15);
    fNodeEq[22] = gpu_f_eq(rhoW3, -ux3 - uy3 + uz3, p1_muu15);
    fNodeEq[23] = gpu_f_eq(rhoW3, ux3 - uy3 + uz3, p1_muu15);
    fNodeEq[24] = gpu_f_eq(rhoW3, -ux3 + uy3 - uz3, p1_muu15);
    fNodeEq[25] = gpu_f_eq(rhoW3, -ux3 + uy3 + uz3, p1_muu15);
    fNodeEq[26] = gpu_f_eq(rhoW3, ux3 - uy3 - uz3, p1_muu15);
#endif

// CALCULATE NON-EQUILIBRIUM POPULATIONS
#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        fNodeNeq[i] = rhoVar * 1.5 * w[i] *
                          (((cx[i] * cx[i] - cs2) * (pixx - pixx_eq) + //Q-iab*(m_ab - m_ab^eq)
                        2 * (cx[i] * cy[i])       * (pixy - pixy_eq) +
                        2 * (cx[i] * cz[i])       * (pixz - pixz_eq) +
                            (cy[i] * cy[i] - cs2) * (piyy - piyy_eq) +
                        2 * (cy[i] * cz[i])       * (piyz - piyz_eq) +
                            (cz[i] * cz[i] - cs2) * (pizz - pizz_eq)) -
                       cs2 * (cx[i] * FX + cy[i] * FY + cz[i] * FZ)); //force term
    }

    //CALCULATE COLLISION POPULATIONS
    pop[ 0] = fNodeEq[ 0] + fNodeNeq[ 0];
    pop[ 1] = fNodeEq[ 1] + fNodeNeq[ 1];
    pop[ 2] = fNodeEq[ 2] + fNodeNeq[ 2];
    pop[ 3] = fNodeEq[ 3] + fNodeNeq[ 3];
    pop[ 4] = fNodeEq[ 4] + fNodeNeq[ 4];
    pop[ 5] = fNodeEq[ 5] + fNodeNeq[ 5];
    pop[ 6] = fNodeEq[ 6] + fNodeNeq[ 6];
    pop[ 7] = fNodeEq[ 7] + fNodeNeq[ 7];
    pop[ 8] = fNodeEq[ 8] + fNodeNeq[ 8];
    pop[ 9] = fNodeEq[ 9] + fNodeNeq[ 9];
    pop[10] = fNodeEq[10] + fNodeNeq[10];
    pop[11] = fNodeEq[11] + fNodeNeq[11];
    pop[12] = fNodeEq[12] + fNodeNeq[12];
    pop[13] = fNodeEq[13] + fNodeNeq[13];
    pop[14] = fNodeEq[14] + fNodeNeq[14];
    pop[15] = fNodeEq[15] + fNodeNeq[15];
    pop[16] = fNodeEq[16] + fNodeNeq[16];
    pop[17] = fNodeEq[17] + fNodeNeq[17];
    pop[18] = fNodeEq[18] + fNodeNeq[18];
#ifdef D3Q27
    pop[19] = fNodeEq[19] + fNodeNeq[19];
    pop[20] = fNodeEq[20] + fNodeNeq[20];
    pop[21] = fNodeEq[21] + fNodeNeq[21];
    pop[22] = fNodeEq[22] + fNodeNeq[22];
    pop[23] = fNodeEq[23] + fNodeNeq[23];
    pop[24] = fNodeEq[24] + fNodeNeq[24];
    pop[25] = fNodeEq[25] + fNodeNeq[25];
    pop[26] = fNodeEq[26] + fNodeNeq[26];
#endif

    

    /* compute rho and u */

    #ifdef D3Q19
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16]) + 0.5 * FX) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18]) + 0.5 * FY) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17]) + 0.5 * FZ) * invRho;
    #endif
    #ifdef D3Q27
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;
    #endif

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
    gpuInterfaceSpread(threadIdx,blockIdx,pop,fGhostX_0, fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1);
}
