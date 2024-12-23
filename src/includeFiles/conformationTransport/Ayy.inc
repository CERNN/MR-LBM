dfloat AyyNode[GQ];

dfloat invAyy = 1/AyyVar;
dfloat Ayy_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Ayy_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Ayy_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

dfloat Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
dfloat Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
dfloat Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

#include COLREC_AYY_RECONSTRUCTION

__syncthreads();

//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = AyyNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = AyyNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = AyyNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = AyyNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = AyyNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = AyyNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = AyyNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = AyyNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = AyyNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = AyyNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = AyyNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = AyyNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = AyyNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = AyyNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = AyyNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = AyyNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = AyyNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = AyyNode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
AyyNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
AyyNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
AyyNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
AyyNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
AyyNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
AyyNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
AyyNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
AyyNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
AyyNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
AyyNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
AyyNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
AyyNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
AyyNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
AyyNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
AyyNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
AyyNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
AyyNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
AyyNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Ayy.inc"

if(nodeType != BULK){
    #include CASE_AYY_BC_DEF
}else{
    AyyVar = AyyNode[0] + AyyNode[1] + AyyNode[2] + AyyNode[3] + AyyNode[4] + AyyNode[5] + AyyNode[6] + AyyNode[7] + AyyNode[8] + AyyNode[9] + AyyNode[10] + AyyNode[11] + AyyNode[12] + AyyNode[13] + AyyNode[14] + AyyNode[15] + AyyNode[16] + AyyNode[17] + AyyNode[18];
    AyyVar = AyyVar; // + T_Q_INTERNAL_D_Cp;
    invAyy= 1.0/AyyVar;

    Ayy_qx_t30 = F_M_I_SCALE*((AyyNode[1] - AyyNode[2] + AyyNode[7] - AyyNode[ 8] + AyyNode[ 9] - AyyNode[10] + AyyNode[13] - AyyNode[14] + AyyNode[15] - AyyNode[16]));
    Ayy_qy_t30 = F_M_I_SCALE*((AyyNode[3] - AyyNode[4] + AyyNode[7] - AyyNode[ 8] + AyyNode[11] - AyyNode[12] + AyyNode[14] - AyyNode[13] + AyyNode[17] - AyyNode[18]));
    Ayy_qz_t30 = F_M_I_SCALE*((AyyNode[5] - AyyNode[6] + AyyNode[9] - AyyNode[10] + AyyNode[11] - AyyNode[12] + AyyNode[16] - AyyNode[15] + AyyNode[18] - AyyNode[17]));
}

Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);