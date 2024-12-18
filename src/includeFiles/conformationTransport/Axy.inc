dfloat AxyNode[GQ];

dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat invAxy = 1/AxyVar;
dfloat Axy_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Axy_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Axy_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

//dfloat udx_t30 = 0.0;
//dfloat udy_t30 = 0.0;
//dfloat udz_t30 = 0.0;

#include COLREC_AXY_RECONSTRUCTION

__syncthreads();

//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = AxyNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = AxyNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = AxyNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = AxyNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = AxyNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = AxyNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = AxyNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = AxyNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = AxyNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = AxyNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = AxyNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = AxyNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = AxyNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = AxyNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = AxyNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = AxyNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = AxyNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = AxyNode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
AxyNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
AxyNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
AxyNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
AxyNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
AxyNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
AxyNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
AxyNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
AxyNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
AxyNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
AxyNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
AxyNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
AxyNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
AxyNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
AxyNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
AxyNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
AxyNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
AxyNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
AxyNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Axy.inc"

if(nodeType != BULK){
    #include CASE_AXY_BC_DEF
}else{
    AxyVar = AxyNode[0] + AxyNode[1] + AxyNode[2] + AxyNode[3] + AxyNode[4] + AxyNode[5] + AxyNode[6] + AxyNode[7] + AxyNode[8] + AxyNode[9] + AxyNode[10] + AxyNode[11] + AxyNode[12] + AxyNode[13] + AxyNode[14] + AxyNode[15] + AxyNode[16] + AxyNode[17] + AxyNode[18];
    AxyVar = AxyVar; // + T_Q_INTERNAL_D_Cp;
    invAxy= 1.0/AxyVar;

    Axy_qx_t30 = F_M_I_SCALE*((AxyNode[1] - AxyNode[2] + AxyNode[7] - AxyNode[ 8] + AxyNode[ 9] - AxyNode[10] + AxyNode[13] - AxyNode[14] + AxyNode[15] - AxyNode[16]));
    Axy_qy_t30 = F_M_I_SCALE*((AxyNode[3] - AxyNode[4] + AxyNode[7] - AxyNode[ 8] + AxyNode[11] - AxyNode[12] + AxyNode[14] - AxyNode[13] + AxyNode[17] - AxyNode[18]));
    Axy_qz_t30 = F_M_I_SCALE*((AxyNode[5] - AxyNode[6] + AxyNode[9] - AxyNode[10] + AxyNode[11] - AxyNode[12] + AxyNode[16] - AxyNode[15] + AxyNode[18] - AxyNode[17]));
}

