
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
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Axx.inc"

if(nodeType != BULK){
    #include CASE_AXX_BC_DEF
}else{
    AxxVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
    AxxVar = AxxVar; // + T_Q_INTERNAL_D_Cp;
    invAxx= 1.0/AxxVar;

    Axx_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
    Axx_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
    Axx_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
}

Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);
