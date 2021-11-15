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

    const unsigned short int tx = threadIdx.x;
    const unsigned short int ty = threadIdx.y;
    const unsigned short int tz = threadIdx.z;

    const unsigned short int bx = blockIdx.x;
    const unsigned short int by = blockIdx.y;
    const unsigned short int bz = blockIdx.z;

    // Load moments from global memory

    dfloat rhoVar = fMom[idxMom(tx, ty, tz, 0, bx, by, bz)];
    dfloat uxVar  = fMom[idxMom(tx, ty, tz, 1, bx, by, bz)];
    dfloat uyVar  = fMom[idxMom(tx, ty, tz, 2, bx, by, bz)];
    dfloat uzVar  = fMom[idxMom(tx, ty, tz, 3, bx, by, bz)];
    dfloat pixx   = fMom[idxMom(tx, ty, tz, 4, bx, by, bz)];
    dfloat pixy   = fMom[idxMom(tx, ty, tz, 5, bx, by, bz)];
    dfloat pixz   = fMom[idxMom(tx, ty, tz, 6, bx, by, bz)];
    dfloat piyy   = fMom[idxMom(tx, ty, tz, 7, bx, by, bz)];
    dfloat piyz   = fMom[idxMom(tx, ty, tz, 8, bx, by, bz)];
    dfloat pizz   = fMom[idxMom(tx, ty, tz, 9, bx, by, bz)];

    dfloat fNodeEq[Q];
    dfloat fNodeNeq[Q];
    dfloat fPop[Q];

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
                        2 * (cx[i] * cy[i]) * (pixy - pixy_eq) +
                        2 * (cx[i] * cz[i]) * (pixz - pixz_eq) +
                        (cy[i] * cy[i] - cs2) * (piyy - piyy_eq) +
                        2 * (cy[i] * cz[i]) * (piyz - piyz_eq) +
                        (cz[i] * cz[i] - cs2) * (pizz - pizz_eq)) -
                       cs2 * (cx[i] * FX + cy[i] * FY + cz[i] * FZ)); //force term
    }

    //CALCULATE COLLISION POPULATIONS
    fPop[0] = fNodeEq[0] + fNodeNeq[0];
    fPop[1] = fNodeEq[1] + fNodeNeq[1];
    fPop[2] = fNodeEq[2] + fNodeNeq[2];
    fPop[3] = fNodeEq[3] + fNodeNeq[3];
    fPop[4] = fNodeEq[4] + fNodeNeq[4];
    fPop[5] = fNodeEq[5] + fNodeNeq[5];
    fPop[6] = fNodeEq[6] + fNodeNeq[6];
    fPop[7] = fNodeEq[7] + fNodeNeq[7];
    fPop[8] = fNodeEq[8] + fNodeNeq[8];
    fPop[9] = fNodeEq[9] + fNodeNeq[9];
    fPop[10] = fNodeEq[10] + fNodeNeq[10];
    fPop[11] = fNodeEq[11] + fNodeNeq[11];
    fPop[12] = fNodeEq[12] + fNodeNeq[12];
    fPop[13] = fNodeEq[13] + fNodeNeq[13];
    fPop[14] = fNodeEq[14] + fNodeNeq[14];
    fPop[15] = fNodeEq[15] + fNodeNeq[15];
    fPop[16] = fNodeEq[16] + fNodeNeq[16];
    fPop[17] = fNodeEq[17] + fNodeNeq[17];
    fPop[18] = fNodeEq[18] + fNodeNeq[18];
#ifdef D3Q27
    fPop[19] = fNodeEq[19] + fNodeNeq[19];
    fPop[20] = fNodeEq[20] + fNodeNeq[20];
    fPop[21] = fNodeEq[21] + fNodeNeq[21];
    fPop[22] = fNodeEq[22] + fNodeNeq[22];
    fPop[23] = fNodeEq[23] + fNodeNeq[23];
    fPop[24] = fNodeEq[24] + fNodeNeq[24];
    fPop[25] = fNodeEq[25] + fNodeNeq[25];
    fPop[26] = fNodeEq[26] + fNodeNeq[26];
#endif

    __shared__ dfloat stream_population[BLOCK_LBM_SIZE * Q];

//save populations in shared memory
#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        stream_population[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, i)] = fPop[i];
    }

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */

    const unsigned short int xp1 = (tx + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (tx - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (ty + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (ty - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (tz + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (tz - 1 + BLOCK_NZ) % BLOCK_NZ;

    pop[1] = stream_population[idxPopBlock(xp1, ty, tz, 0)];
    pop[2] = stream_population[idxPopBlock(xm1, ty, tz, 1)];
    pop[3] = stream_population[idxPopBlock(tx, yp1, tz, 2)];
    pop[4] = stream_population[idxPopBlock(tx, ym1, tz, 3)];
    pop[5] = stream_population[idxPopBlock(tx, ty, zp1, 4)];
    pop[6] = stream_population[idxPopBlock(tx, ty, zm1, 5)];
    pop[7] = stream_population[idxPopBlock(xp1, yp1, tz, 6)];
    pop[8] = stream_population[idxPopBlock(xm1, ym1, tz, 7)];
    pop[9] = stream_population[idxPopBlock(xp1, ty, zp1, 8)];
    pop[10] = stream_population[idxPopBlock(xm1, ty, zm1, 9)];
    pop[11] = stream_population[idxPopBlock(tx, yp1, zp1, 10)];
    pop[12] = stream_population[idxPopBlock(tx, ym1, zm1, 11)];
    pop[13] = stream_population[idxPopBlock(xp1, ym1, tz, 12)];
    pop[14] = stream_population[idxPopBlock(xm1, yp1, tz, 13)];
    pop[15] = stream_population[idxPopBlock(xp1, ty, zm1, 14)];
    pop[16] = stream_population[idxPopBlock(xm1, ty, zp1, 15)];
    pop[17] = stream_population[idxPopBlock(tx, yp1, zm1, 16)];
    pop[18] = stream_population[idxPopBlock(tx, ym1, zp1, 17)];

    /* load pop from global in cover nodes */

    if (threadIdx.x == 0)
    {
        pop[1] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[7] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[9] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[13] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[15] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    }
    else if (threadIdx.x == blockDim.x - 1)
    {
        pop[2] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[8] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[10] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[14] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[16] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    }
    if (threadIdx.y == 0)
    {
        pop[3] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[7] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[11] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[14] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[17] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    }
    else if (threadIdx.y == blockDim.y - 1)
    {
        pop[4] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[8] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[12] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[13] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[18] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    }
    if (threadIdx.z == 0)
    {
        pop[5] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[9] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[11] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[16] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[18] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    }
    else if (threadIdx.z == blockDim.z - 1)
    {
        pop[6] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[10] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[12] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[15] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[17] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    }


    #ifdef D3Q19
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        dfloat invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16]) + 0.5 * FX) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18]) + 0.5 * FY) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17]) + 0.5 * FZ) * invRho;
    #endif
    #ifdef D3Q27
        rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        const dfloat invRho = 1 / rhoVar;
        uxVar = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26]) - (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * fxVar) * invRho;
        uyVar = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[4] + pop[8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * fyVar) * invRho;
        uzVar = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * fzVar) * invRho;
    #endif

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
    const dfloat rhoW3 = rhoVar * W3;
    const dfloat W3t9d2 = W3 * 9 / 2;
#endif
    ux3 = 3 * uxVar;
    uy3 = 3 * uyVar;
    uz3 = 3 * uzVar;

    // Calculate equilibrium fNodeEq
    fNodeEq[0] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNodeEq[1] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNodeEq[2] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNodeEq[3] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNodeEq[4] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNodeEq[5] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNodeEq[6] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNodeEq[7] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNodeEq[8] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNodeEq[9] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
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
                        2 * (cx[i] * cy[i]) * (pixy - pixy_eq) +
                        2 * (cx[i] * cz[i]) * (pixz - pixz_eq) +
                        (cy[i] * cy[i] - cs2) * (piyy - piyy_eq) +
                        2 * (cy[i] * cz[i]) * (piyz - piyz_eq) +
                        (cz[i] * cz[i] - cs2) * (pizz - pizz_eq)) -
                       cs2 * (cx[i] * FX + cy[i] * FY + cz[i] * FZ)); //force term
    }

    //CALCULATE COLLISION POPULATIONS
    pop[0] = fNodeEq[0] + fNodeNeq[0];
    pop[1] = fNodeEq[1] + fNodeNeq[1];
    pop[2] = fNodeEq[2] + fNodeNeq[2];
    pop[3] = fNodeEq[3] + fNodeNeq[3];
    pop[4] = fNodeEq[4] + fNodeNeq[4];
    pop[5] = fNodeEq[5] + fNodeNeq[5];
    pop[6] = fNodeEq[6] + fNodeNeq[6];
    pop[7] = fNodeEq[7] + fNodeNeq[7];
    pop[8] = fNodeEq[8] + fNodeNeq[8];
    pop[9] = fNodeEq[9] + fNodeNeq[9];
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

    rhoVar = pop[0] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9];
    uxVar = (-pop[10] + pop[13] - pop[14] + pop[15] - pop[16] + pop[1] - pop[2] + pop[7] - pop[8] + pop[9])/rhoVar;
    uyVar = (pop[11] - pop[12] - pop[13] + pop[14] + pop[17] - pop[18] + pop[3] - pop[4] + pop[7] - pop[8])/rhoVar;
    uzVar = (-pop[10] + pop[11] - pop[12] - pop[15] + pop[16] - pop[17] + pop[18] + pop[5] - pop[6] + pop[9])/rhoVar;

    /* write to global mom */

    fMom[idxMom(tx, ty, tz, 0, bx, by, bz)] = rhoVar;
    fMom[idxMom(tx, ty, tz, 1, bx, by, bz)] = uxVar;
    fMom[idxMom(tx, ty, tz, 2, bx, by, bz)] = uyVar;
    fMom[idxMom(tx, ty, tz, 3, bx, by, bz)] = uzVar;
    fMom[idxMom(tx, ty, tz, 4, bx, by, bz)] = pixx;
    fMom[idxMom(tx, ty, tz, 5, bx, by, bz)] = pixy;
    fMom[idxMom(tx, ty, tz, 6, bx, by, bz)] = pixz;
    fMom[idxMom(tx, ty, tz, 7, bx, by, bz)] = piyy;
    fMom[idxMom(tx, ty, tz, 8, bx, by, bz)] = piyz;
    fMom[idxMom(tx, ty, tz, 9, bx, by, bz)] = pizz;

    /* write to global pop */

    if(ty == 0)             {//s
        fGhostY_0[idxPopY(tx,tz,0,(bx),(by),(bz))] = pop[ 4];
        fGhostY_0[idxPopY(tx,tz,1,(bx),(by),(bz))] = pop[ 8];
        fGhostY_0[idxPopY(tx,tz,2,(bx),(by),(bz))] = pop[12];
        fGhostY_0[idxPopY(tx,tz,3,(bx),(by),(bz))] = pop[13];
        fGhostY_0[idxPopY(tx,tz,4,(bx),(by),(bz))] = pop[18];
    }else if(ty == (BLOCK_NY-1))  {//n
        fGhostY_1[idxPopY(tx,tz,0,(bx),(by),(bz))] = pop[ 3];
        fGhostY_1[idxPopY(tx,tz,1,(bx),(by),(bz))] = pop[ 7];
        fGhostY_1[idxPopY(tx,tz,2,(bx),(by),(bz))] = pop[11];
        fGhostY_1[idxPopY(tx,tz,3,(bx),(by),(bz))] = pop[14];
        fGhostY_1[idxPopY(tx,tz,4,(bx),(by),(bz))] = pop[17];
    }
    if(tx == 0)             {//w
        fGhostX_0[idxPopX(ty,tz,0,(bx),(by),(bz))] = pop[ 2];
        fGhostX_0[idxPopX(ty,tz,1,(bx),(by),(bz))] = pop[ 8];
        fGhostX_0[idxPopX(ty,tz,2,(bx),(by),(bz))] = pop[10];
        fGhostX_0[idxPopX(ty,tz,3,(bx),(by),(bz))] = pop[14];
        fGhostX_0[idxPopX(ty,tz,4,(bx),(by),(bz))] = pop[16];
    }else if(tx == (BLOCK_NX-1))  {//e
        fGhostX_1[idxPopX(ty,tz,0,(bx),(by),(bz))] = pop[ 1];
        fGhostX_1[idxPopX(ty,tz,1,(bx),(by),(bz))] = pop[ 7];
        fGhostX_1[idxPopX(ty,tz,2,(bx),(by),(bz))] = pop[ 9];
        fGhostX_1[idxPopX(ty,tz,3,(bx),(by),(bz))] = pop[13];
        fGhostX_1[idxPopX(ty,tz,4,(bx),(by),(bz))] = pop[15];
    }
    if(tz == 0)             {//b
        fGhostZ_0[idxPopZ(tx,ty,0,(bx),(by),(bz))] = pop[ 6];
        fGhostZ_0[idxPopZ(tx,ty,1,(bx),(by),(bz))] = pop[10];
        fGhostZ_0[idxPopZ(tx,ty,2,(bx),(by),(bz))] = pop[12];
        fGhostZ_0[idxPopZ(tx,ty,3,(bx),(by),(bz))] = pop[15];
        fGhostZ_0[idxPopZ(tx,ty,4,(bx),(by),(bz))] = pop[17];
    }else if(tz == (BLOCK_NZ-1))  {//f
        fGhostZ_1[idxPopZ(tx,ty,0,(bx),(by),(bz))] = pop[ 5];
        fGhostZ_1[idxPopZ(tx,ty,1,(bx),(by),(bz))] = pop[ 9];
        fGhostZ_1[idxPopZ(tx,ty,2,(bx),(by),(bz))] = pop[11];
        fGhostZ_1[idxPopZ(tx,ty,3,(bx),(by),(bz))] = pop[16];
        fGhostZ_1[idxPopZ(tx,ty,4,(bx),(by),(bz))] = pop[18];
    } 
}
