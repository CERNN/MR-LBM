#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(
    dfloat *fMom, unsigned char *dNodeType,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1,
    dfloat *gGhostX_0, dfloat *gGhostX_1,
    dfloat *gGhostY_0, dfloat *gGhostY_1,
    dfloat *gGhostZ_0, dfloat *gGhostZ_1,
    #ifdef DENSITY_CORRECTION
    dfloat *d_mean_rho,
    #endif
    unsigned int step)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat pop[Q];

    // Load moments from global memory

    //rho'
    unsigned char nodeType = dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat rhoVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat ux_t30     = 3.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30     = 3.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30     = 3.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xx_t45   = 4.5*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xy_t90   = 9.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xz_t90   = 9.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_yy_t45   = 4.5*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_yz_t90   = 9.0*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_zz_t45   = 4.5*fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)];

    #ifdef NON_NEWTONIAN_FLUID
    dfloat omegaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 10, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat t_omegaVar = 1 - omegaVar;
    dfloat tt_omegaVar = 1 - 0.5*omegaVar;
    dfloat omegaVar_d2 = omegaVar / 2.0;
    dfloat tt_omega_t3 = tt_omegaVar * 3.0;
    #else
    dfloat omegaVar = OMEGA;
    #endif

    
    //calculate post collision populations
    dfloat multiplyTerm;
    multiplyTerm = rhoVar * W0;
    dfloat pics2 = 1.0 - cs2 * (m_xx_t45 + m_yy_t45 + m_zz_t45);

    pop[ 0] = multiplyTerm * (pics2);
    multiplyTerm = rhoVar * W1;
    pop[ 1] = multiplyTerm * (pics2 + ux_t30 + m_xx_t45);
    pop[ 2] = multiplyTerm * (pics2 - ux_t30 + m_xx_t45);
    pop[ 3] = multiplyTerm * (pics2 + uy_t30 + m_yy_t45);
    pop[ 4] = multiplyTerm * (pics2 - uy_t30 + m_yy_t45);
    pop[ 5] = multiplyTerm * (pics2 + uz_t30 + m_zz_t45);
    pop[ 6] = multiplyTerm * (pics2 - uz_t30 + m_zz_t45);
    multiplyTerm = rhoVar * W2;
    pop[ 7] = multiplyTerm * (pics2 +ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 + m_xy_t90);
    pop[ 8] = multiplyTerm * (pics2 -ux_t30 + m_xx_t45 - uy_t30 + m_yy_t45 + m_xy_t90);
    pop[ 9] = multiplyTerm * (pics2 +ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 + m_xz_t90);
    pop[10] = multiplyTerm * (pics2 -ux_t30 + m_xx_t45 - uz_t30 + m_zz_t45 + m_xz_t90);
    pop[11] = multiplyTerm * (pics2 +uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 + m_yz_t90);
    pop[12] = multiplyTerm * (pics2 -uy_t30 + m_yy_t45 - uz_t30 + m_zz_t45 + m_yz_t90);
    pop[13] = multiplyTerm * (pics2 +ux_t30 - uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
    pop[14] = multiplyTerm * (pics2 -ux_t30 + uy_t30 + m_xx_t45 + m_yy_t45 - m_xy_t90);
    pop[15] = multiplyTerm * (pics2 +ux_t30 - uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
    pop[16] = multiplyTerm * (pics2 -ux_t30 + uz_t30 + m_xx_t45 + m_zz_t45 - m_xz_t90);
    pop[17] = multiplyTerm * (pics2 +uy_t30 - uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);
    pop[18] = multiplyTerm * (pics2 -uy_t30 + uz_t30 + m_yy_t45 + m_zz_t45 - m_yz_t90);   
    #ifdef D3Q27
    multiplyTerm = rhoVar * W3;
    pop[19] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90));
    pop[20] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90));
    pop[21] = multiplyTerm * (pics2 + ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90));
    pop[22] = multiplyTerm * (pics2 - ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90));
    pop[23] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90));
    pop[24] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90));
    pop[25] = multiplyTerm * (pics2 - ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90));
    pop[26] = multiplyTerm * (pics2 + ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90));
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


   
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;


    #include "interfaceInclude/popLoad"

    //NOTE : STREAMING DONE, APPLY BOUNDARY CONDITION AND COMPUTE POST STREAMING MOMENTS
    #ifdef BC_POPULATION_BASED

        if (nodeType){
            #include BC_PATH
        }
            //calculate streaming moments
        #ifdef D3Q19
            //equation3
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
            dfloat invRho = 1 / rhoVar;
            //equation4 + force correction
            ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) + 0.5 * FX) * invRho;
            uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) + 0.5 * FY) * invRho;
            uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) + 0.5 * FZ) * invRho;


            //equation5
            m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16])* invRho - cs2;
            m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14])* invRho;
            m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16])* invRho;
            m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18])* invRho - cs2;
            m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18])* invRho;
            m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18])* invRho - cs2;


        #endif
        #ifdef D3Q27
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
            dfloat invRho = 1 / rhoVar;
            ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
            uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
            uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;

            m_xx_t45 = ( (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
            m_xy_t90 = (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) )* invRho;
            m_xz_t90 = (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) )* invRho;
            m_yy_t45 = ( (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
            m_yz_t90 = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])))* invRho;
            m_zz_t45 = ( (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
        #endif
    #endif

    #ifdef BC_MOMENT_BASED
        dfloat invRho;
        if(nodeType != BULK){
            #include BC_PATH

            invRho = 1.0 / rho;
        }else{

            //calculate streaming moments
            #ifdef D3Q19
                //equation3
                rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
                invRho = 1 / rhoVar;
                //equation4 + force correction
                ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[ 8] + pop[ 9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) + 0.5 * FX) * invRho;
                uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[ 8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) + 0.5 * FY) * invRho;
                uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) + 0.5 * FZ) * invRho;

                //equation5
                m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16])* invRho - cs2;
                m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14])* invRho;
                m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16])* invRho;
                m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18])* invRho - cs2;
                m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18])* invRho;
                m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18])* invRho - cs2;


            #endif
            #ifdef D3Q27
                rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
                invRho = 1 / rhoVar;
                ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * FX) * invRho;
                uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * FY) * invRho;
                uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * FZ) * invRho;

                m_xx_t45 = ( (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
                m_xy_t90 = (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) );
                m_xz_t90 = (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) );
                m_yy_t45 = ( (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
                m_yz_t90 = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])));
                m_zz_t45 = ( (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]));
            #endif
        }
    #endif // BC_MOMENT_BASED


    // MOMENTS DETERMINED, COMPUTE OMEGA IF NON-NEWTONIAN FLUID
    #ifdef NON_NEWTONIAN_FLUID

    const dfloat S_XX = rhoVar * (m_xx_t45 - ux_t30*ux_t30);
    const dfloat S_YY = rhoVar * (m_yy_t45 - uy_t30*uy_t30);
    const dfloat S_ZZ = rhoVar * (m_zz_t45 - uz_t30*uz_t30);
    const dfloat S_XY = rhoVar * (m_xy_t90 - ux_t30*uy_t30);
    const dfloat S_XZ = rhoVar * (m_xz_t90 - ux_t30*uz_t30);
    const dfloat S_YZ = rhoVar * (m_yz_t90 - uy_t30*uz_t30);

    const dfloat uFxxd2 = ux_t30*FX; // d2 = uFxx Divided by two
    const dfloat uFyyd2 = uy_t30*FY;
    const dfloat uFzzd2 = uz_t30*FZ;
    const dfloat uFxyd2 = (ux_t30*FY + uy_t30*FX) / 2;
    const dfloat uFxzd2 = (ux_t30*FZ + uz_t30*FX) / 2;
    const dfloat uFyzd2 = (uy_t30*FZ + uz_t30*FY) / 2;

    const dfloat auxStressMag = sqrt(0.5 * (
        (S_XX + uFxxd2) * (S_XX + uFxxd2) +(S_YY + uFyyd2) * (S_YY + uFyyd2) + (S_ZZ + uFzzd2) * (S_ZZ + uFzzd2) +
        2 * ((S_XY + uFxyd2) * (S_XY + uFxyd2) + (S_XZ + uFxzd2) * (S_XZ + uFxzd2) + (S_YZ + uFyzd2) * (S_YZ + uFyzd2))));

    /*
    dfloat eta = (1.0/omegaVar - 0.5) / 3.0;
    dfloat gamma_dot = (1 - 0.5 * (omegaVar)) * auxStressMag / eta;
    eta = VISC + S_Y/gamma_dot;
    omegaVar = omegaVar;// 1.0 / (0.5 + 3.0 * eta);
    */

    omegaVar = calcOmega(omegaVar, auxStressMag);

    t_omegaVar = 1 - omegaVar;
    tt_omegaVar = 1 - 0.5*omegaVar;
    omegaVar_d2 = omegaVar / 2.0;
    tt_omega_t3 = tt_omegaVar * 3.0;
    #endif//  NON_NEWTONIAN_FLUID


    // COLLIDE

    //Collide Moments
    // multiply moments by as2 -- as4/2 -- as4 - add correction to m_alpha_beta
    #ifndef HIGH_ORDER_COLLISION
        ux_t30 = 3.0 * ux_t30;
        uy_t30 = 3.0 * uy_t30;
        uz_t30 = 3.0 * uz_t30;

        m_xx_t45 = 4.5 * (m_xx_t45);
        m_xy_t90 = 9.0 * (m_xy_t90);
        m_xz_t90 = 9.0 * (m_xz_t90);
        m_yy_t45 = 4.5 * (m_yy_t45);
        m_yz_t90 = 9.0 * (m_yz_t90);
        m_zz_t45 = 4.5 * (m_zz_t45);

        #ifdef DENSITY_CORRECTION
            //printf("%f ",d_mean_rho[0]-1) ;
            rhoVar -= (d_mean_rho[0]-1e-7) ;
            invRho = 1/rhoVar;
        #endif // DENSITY_CORRECTION
        #ifdef NON_NEWTONIAN_FLUID
            dfloat invRho_mt15 = -1.5*invRho;
            ux_t30 = (t_omegaVar * (ux_t30 + invRho_mt15 * FX ) + omegaVar * ux_t30 + tt_omega_t3 * FX);
            uy_t30 = (t_omegaVar * (uy_t30 + invRho_mt15 * FY ) + omegaVar * uy_t30 + tt_omega_t3 * FY);
            uz_t30 = (t_omegaVar * (uz_t30 + invRho_mt15 * FZ ) + omegaVar * uz_t30 + tt_omega_t3 * FZ);
            
            //equation 90
            m_xx_t45 = (t_omegaVar * m_xx_t45  +   omegaVar_d2 * ux_t30 * ux_t30  - invRho_mt15 * tt_omegaVar * (FX * ux_t30 + FX * ux_t30));
            m_yy_t45 = (t_omegaVar * m_yy_t45  +   omegaVar_d2 * uy_t30 * uy_t30  - invRho_mt15 * tt_omegaVar * (FY * uy_t30 + FY * uy_t30));
            m_zz_t45 = (t_omegaVar * m_zz_t45  +   omegaVar_d2 * uz_t30 * uz_t30  - invRho_mt15 * tt_omegaVar * (FZ * uz_t30 + FZ * uz_t30));

            m_xy_t90 = (t_omegaVar * m_xy_t90  +   omegaVar * ux_t30 * uy_t30    +    tt_omega_t3 *invRho* (FX * uy_t30 + FY * ux_t30));
            m_xz_t90 = (t_omegaVar * m_xz_t90  +   omegaVar * ux_t30 * uz_t30    +    tt_omega_t3 *invRho* (FX * uz_t30 + FZ * ux_t30));
            m_yz_t90 = (t_omegaVar * m_yz_t90  +   omegaVar * uy_t30 * uz_t30    +    tt_omega_t3 *invRho* (FY * uz_t30 + FZ * uy_t30));
        #endif // NON_NEWTONIAN_FLUID
        #ifndef NON_NEWTONIAN_FLUID 
            dfloat invRho_mt15 = -1.5*invRho;
            ux_t30 = (T_OMEGA * (ux_t30 + invRho_mt15 * FX ) + OMEGA * ux_t30 + TT_OMEGA_T3 * FX);
            uy_t30 = (T_OMEGA * (uy_t30 + invRho_mt15 * FY ) + OMEGA * uy_t30 + TT_OMEGA_T3 * FY);
            uz_t30 = (T_OMEGA * (uz_t30 + invRho_mt15 * FZ ) + OMEGA * uz_t30 + TT_OMEGA_T3 * FZ);
            
            //equation 90
            m_xx_t45 = (T_OMEGA * m_xx_t45  +   OMEGAd2 * ux_t30 * ux_t30    - invRho_mt15 * TT_OMEGA * (FX * ux_t30 + FX * ux_t30));
            m_yy_t45 = (T_OMEGA * m_yy_t45  +   OMEGAd2 * uy_t30 * uy_t30    - invRho_mt15 * TT_OMEGA * (FY * uy_t30 + FY * uy_t30));
            m_zz_t45 = (T_OMEGA * m_zz_t45  +   OMEGAd2 * uz_t30 * uz_t30    - invRho_mt15 * TT_OMEGA * (FZ * uz_t30 + FZ * uz_t30));

            m_xy_t90 = (T_OMEGA * m_xy_t90  +     OMEGA * ux_t30 * uy_t30    +    TT_OMEGA_T3 *invRho* (FX * uy_t30 + FY * ux_t30));
            m_xz_t90 = (T_OMEGA * m_xz_t90  +     OMEGA * ux_t30 * uz_t30    +    TT_OMEGA_T3 *invRho* (FX * uz_t30 + FZ * ux_t30));
            m_yz_t90 = (T_OMEGA * m_yz_t90  +     OMEGA * uy_t30 * uz_t30    +    TT_OMEGA_T3 *invRho* (FY * uz_t30 + FZ * uy_t30));
        #endif //!_NON_NEWTONIAN_FLUID
    #endif //!_HIGH_ORDER_COLLISION

    //USING HIGH
    #ifdef HIGH_ORDER_COLLISION

    const dfloat TAU_XX = (0.5-1/omegaVar)*cs2*(rhoVar * (m_xx_t45 - ux_t30*ux_t30));
    const dfloat TAU_YY = (0.5-1/omegaVar)*cs2*(rhoVar * (m_yy_t45 - uy_t30*uy_t30));
    const dfloat TAU_ZZ = (0.5-1/omegaVar)*cs2*(rhoVar * (m_zz_t45 - uz_t30*uz_t30));
    const dfloat TAU_XY = (0.5-1/omegaVar)*cs2*(rhoVar * (m_xy_t90 - ux_t30*uy_t30));
    const dfloat TAU_XZ = (0.5-1/omegaVar)*cs2*(rhoVar * (m_xz_t90 - ux_t30*uz_t30));
    const dfloat TAU_YZ = (0.5-1/omegaVar)*cs2*(rhoVar * (m_yz_t90 - uy_t30*uz_t30));



    dfloat feq;
    dfloat fneq;
    for(int i = 0; i<Q ; i++){
        feq = w[i]*(rhoVar*(3*cx[i]*ux_t30 + 3*cy[i]*uy_t30 + 3*cz[i]*uz_t30 + (ux_t30*ux_t30*(9*cx[i]*cx[i] - 3))/2 + (uy_t30*uy_t30*(9*cy[i]*cy[i] - 3))/2 + (uz_t30*uz_t30*(9*cz[i]*cz[i] - 3))/2 + (9*ux_t30*ux_t30*uy_t30*uy_t30*(3*cx[i]*cx[i] - 1)*(3*cy[i]*cy[i] - 1))/4 + (9*ux_t30*ux_t30*uz_t30*uz_t30*(3*cx[i]*cx[i] - 1)*(3*cz[i]*cz[i] - 1))/4 + (9*uy_t30*uy_t30*uz_t30*uz_t30*(3*cy[i]*cy[i] - 1)*(3*cz[i]*cz[i] - 1))/4 + 9*cx[i]*cy[i]*ux_t30*uy_t30 + 9*cx[i]*cz[i]*ux_t30*uz_t30 + 9*cy[i]*cz[i]*uy_t30*uz_t30 + (9*cx[i]*ux_t30*uy_t30*uy_t30*(3*cy[i]*cy[i] - 1))/2 + (9*cy[i]*ux_t30*ux_t30*uy_t30*(3*cx[i]*cx[i] - 1))/2 + (9*cx[i]*ux_t30*uz_t30*uz_t30*(3*cz[i]*cz[i] - 1))/2 + (9*cz[i]*ux_t30*ux_t30*uz_t30*(3*cx[i]*cx[i] - 1))/2 + (9*cy[i]*uy_t30*uz_t30*uz_t30*(3*cz[i]*cz[i] - 1))/2 + (9*cz[i]*uy_t30*uy_t30*uz_t30*(3*cy[i]*cy[i] - 1))/2 + 1));

        fneq = w[i]*((TAU_XX*(9*cx[i]*cx[i] - 3))/2 + (TAU_YY*(9*cy[i]*cy[i] - 3))/2 + (TAU_ZZ*(9*cz[i]*cz[i] - 3))/2 + (9*(3*cx[i]*cx[i] - 1)*(3*cy[i]*cy[i] - 1)*(TAU_YY*ux_t30*ux_t30 + TAU_XX*uy_t30*uy_t30 + 4*TAU_XY*ux_t30*uy_t30))/4 + (9*(3*cx[i]*cx[i] - 1)*(3*cz[i]*cz[i] - 1)*(TAU_ZZ*ux_t30*ux_t30 + TAU_XX*uz_t30*uz_t30 + 4*TAU_XZ*ux_t30*uz_t30))/4 + (9*(3*cy[i]*cy[i] - 1)*(3*cz[i]*cz[i] - 1)*(TAU_ZZ*uy_t30*uy_t30 + TAU_YY*uz_t30*uz_t30 + 4*TAU_YZ*uy_t30*uz_t30))/4 + 9*TAU_XY*cx[i]*cy[i] + 9*TAU_XZ*cx[i]*cz[i] + 9*TAU_YZ*cy[i]*cz[i] + (9*cy[i]*(3*cx[i]*cx[i] - 1)*(2*TAU_XY*ux_t30 + TAU_XX*uy_t30))/2 + (9*cx[i]*(3*cy[i]*cy[i] - 1)*(TAU_YY*ux_t30 + 2*TAU_XY*uy_t30))/2 + (9*cz[i]*(3*cx[i]*cx[i] - 1)*(2*TAU_XZ*ux_t30 + TAU_XX*uz_t30))/2 + (9*cx[i]*(3*cz[i]*cz[i] - 1)*(TAU_ZZ*ux_t30 + 2*TAU_XZ*uz_t30))/2 + (9*cz[i]*(3*cy[i]*cy[i] - 1)*(2*TAU_YZ*uy_t30 + TAU_YY*uz_t30))/2 + (9*cy[i]*(3*cz[i]*cz[i] - 1)*(TAU_ZZ*uy_t30 + 2*TAU_YZ*uz_t30))/2);

        pop[i] = feq + omegaVar * fneq;

    }
    rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    invRho = 1 / rhoVar;

    ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[8] + pop[9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) + 0.5 * FX) * invRho;
    uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) + 0.5 * FY) * invRho;
    uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) + 0.5 * FZ) * invRho;

    m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16])* invRho - cs2;
    m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14])* invRho;
    m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16])* invRho;
    m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18])* invRho - cs2;
    m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18])* invRho;
    m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18])* invRho - cs2;


    ux_t30 = 3.0 * ux_t30;
    uy_t30 = 3.0 * uy_t30;
    uz_t30 = 3.0 * uz_t30;

    m_xx_t45 = 4.5*(m_xx_t45);
    m_xy_t90 = 9.0*(m_xy_t90);
    m_xz_t90 = 9.0*(m_xz_t90);
    m_yy_t45 = 4.5*(m_yy_t45);
    m_yz_t90 = 9.0*(m_yz_t90);
    m_zz_t45 = 4.5*(m_zz_t45);



    #endif //HIGH_ORDER_COLLISION



    //calculate post collision populations
    
    multiplyTerm = rhoVar * W0;
    pics2 = 1.0 - cs2 * (m_xx_t45 + m_yy_t45 + m_zz_t45);

    pop[ 0] = multiplyTerm * (pics2);
    multiplyTerm = rhoVar * W1;
    pop[ 1] = multiplyTerm * (pics2 + ux_t30 + m_xx_t45);
    pop[ 2] = multiplyTerm * (pics2 - ux_t30 + m_xx_t45);
    pop[ 3] = multiplyTerm * (pics2 + uy_t30 + m_yy_t45);
    pop[ 4] = multiplyTerm * (pics2 - uy_t30 + m_yy_t45);
    pop[ 5] = multiplyTerm * (pics2 + uz_t30 + m_zz_t45);
    pop[ 6] = multiplyTerm * (pics2 - uz_t30 + m_zz_t45);
    multiplyTerm = rhoVar * W2;
    pop[ 7] = multiplyTerm * (pics2 + ( ux_t30 + uy_t30) + (m_xx_t45 + m_yy_t45) + m_xy_t90);
    pop[ 8] = multiplyTerm * (pics2 + (-ux_t30 - uy_t30) + (m_xx_t45 + m_yy_t45) + m_xy_t90);
    pop[ 9] = multiplyTerm * (pics2 + ( ux_t30 + uz_t30) + (m_xx_t45 + m_zz_t45) + m_xz_t90);
    pop[10] = multiplyTerm * (pics2 + (-ux_t30 - uz_t30) + (m_xx_t45 + m_zz_t45) + m_xz_t90);
    pop[11] = multiplyTerm * (pics2 + ( uy_t30 + uz_t30) + (m_yy_t45 + m_zz_t45) + m_yz_t90);
    pop[12] = multiplyTerm * (pics2 + (-uy_t30 - uz_t30) + (m_yy_t45 + m_zz_t45) + m_yz_t90);
    pop[13] = multiplyTerm * (pics2 + ( ux_t30 - uy_t30) + (m_xx_t45 + m_yy_t45) - m_xy_t90);
    pop[14] = multiplyTerm * (pics2 + (-ux_t30 + uy_t30) + (m_xx_t45 + m_yy_t45) - m_xy_t90);
    pop[15] = multiplyTerm * (pics2 + ( ux_t30 - uz_t30) + (m_xx_t45 + m_zz_t45) - m_xz_t90);
    pop[16] = multiplyTerm * (pics2 + (-ux_t30 + uz_t30) + (m_xx_t45 + m_zz_t45) - m_xz_t90);
    pop[17] = multiplyTerm * (pics2 + ( uy_t30 - uz_t30) + (m_yy_t45 + m_zz_t45) - m_yz_t90);
    pop[18] = multiplyTerm * (pics2 + (-uy_t30 + uz_t30) + (m_yy_t45 + m_zz_t45) - m_yz_t90);   
    #ifdef D3Q27
    multiplyTerm = rhoVar * W3;
    pop[19] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90));
    pop[20] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 + m_xz_t90 + m_yz_t90));
    pop[21] = multiplyTerm * (pics2 + ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90));
    pop[22] = multiplyTerm * (pics2 - ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 + (m_xy_t90 - m_xz_t90 - m_yz_t90));
    pop[23] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90));
    pop[24] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 - m_xz_t90 + m_yz_t90));
    pop[25] = multiplyTerm * (pics2 - ux_t30 + uy_t30 + uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90));
    pop[26] = multiplyTerm * (pics2 + ux_t30 - uy_t30 - uz_t30 + m_xx_t45 + m_yy_t45 + m_zz_t45 - (m_xy_t90 + m_xz_t90 - m_yz_t90));
    #endif //D3Q27
    
    /* write to global mom */

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30/3.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30/3.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30/3.0;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45/4.5;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90/9.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90/9.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45/4.5;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90/9.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45/4.5;
    
    #ifdef NON_NEWTONIAN_FLUID
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 10, blockIdx.x, blockIdx.y, blockIdx.z)] = omegaVar;
    #endif
 
    #include "interfaceInclude/popSave"
}
