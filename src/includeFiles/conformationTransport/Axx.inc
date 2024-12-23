dfloat AxxNode[GQ];

dfloat invAxx = 1/AxxVar;
dfloat Axx_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Axx_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Axx_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

dfloat Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
dfloat Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
dfloat Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

#include COLREC_AXX_RECONSTRUCTION

__syncthreads();

//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = AxxNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = AxxNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = AxxNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = AxxNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = AxxNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = AxxNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = AxxNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = AxxNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = AxxNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = AxxNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = AxxNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = AxxNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = AxxNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = AxxNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = AxxNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = AxxNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = AxxNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = AxxNode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
AxxNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
AxxNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
AxxNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
AxxNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
AxxNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
AxxNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
AxxNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
AxxNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
AxxNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
AxxNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
AxxNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
AxxNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
AxxNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
AxxNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
AxxNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
AxxNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
AxxNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
AxxNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Axx.inc"

if(nodeType != BULK){
    #include CASE_AXX_BC_DEF
}else{
    AxxVar = AxxNode[0] + AxxNode[1] + AxxNode[2] + AxxNode[3] + AxxNode[4] + AxxNode[5] + AxxNode[6] + AxxNode[7] + AxxNode[8] + AxxNode[9] + AxxNode[10] + AxxNode[11] + AxxNode[12] + AxxNode[13] + AxxNode[14] + AxxNode[15] + AxxNode[16] + AxxNode[17] + AxxNode[18];
    AxxVar = AxxVar; // + T_Q_INTERNAL_D_Cp;
    invAxx= 1.0/AxxVar;

    Axx_qx_t30 = F_M_I_SCALE*((AxxNode[1] - AxxNode[2] + AxxNode[7] - AxxNode[ 8] + AxxNode[ 9] - AxxNode[10] + AxxNode[13] - AxxNode[14] + AxxNode[15] - AxxNode[16]));
    Axx_qy_t30 = F_M_I_SCALE*((AxxNode[3] - AxxNode[4] + AxxNode[7] - AxxNode[ 8] + AxxNode[11] - AxxNode[12] + AxxNode[14] - AxxNode[13] + AxxNode[17] - AxxNode[18]));
    Axx_qz_t30 = F_M_I_SCALE*((AxxNode[5] - AxxNode[6] + AxxNode[9] - AxxNode[10] + AxxNode[11] - AxxNode[12] + AxxNode[16] - AxxNode[15] + AxxNode[18] - AxxNode[17]));
}

Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);
