dfloat AxzNode[GQ];

dfloat invAxz = 1/AxzVar;
dfloat Axz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Axz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Axz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

dfloat Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
dfloat Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
dfloat Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

#include COLREC_AXZ_RECONSTRUCTION

__syncthreads();

//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = AxzNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = AxzNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = AxzNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = AxzNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = AxzNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = AxzNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = AxzNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = AxzNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = AxzNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = AxzNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = AxzNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = AxzNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = AxzNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = AxzNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = AxzNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = AxzNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = AxzNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = AxzNode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
AxzNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
AxzNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
AxzNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
AxzNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
AxzNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
AxzNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
AxzNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
AxzNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
AxzNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
AxzNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
AxzNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
AxzNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
AxzNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
AxzNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
AxzNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
AxzNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
AxzNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
AxzNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Axz.inc"

if(nodeType != BULK){
    #include CASE_AXZ_BC_DEF
}else{
    AxzVar = AxzNode[0] + AxzNode[1] + AxzNode[2] + AxzNode[3] + AxzNode[4] + AxzNode[5] + AxzNode[6] + AxzNode[7] + AxzNode[8] + AxzNode[9] + AxzNode[10] + AxzNode[11] + AxzNode[12] + AxzNode[13] + AxzNode[14] + AxzNode[15] + AxzNode[16] + AxzNode[17] + AxzNode[18];
    AxzVar = AxzVar; // + T_Q_INTERNAL_D_Cp;
    invAxz= 1.0/AxzVar;

    Axz_qx_t30 = F_M_I_SCALE*((AxzNode[1] - AxzNode[2] + AxzNode[7] - AxzNode[ 8] + AxzNode[ 9] - AxzNode[10] + AxzNode[13] - AxzNode[14] + AxzNode[15] - AxzNode[16]));
    Axz_qy_t30 = F_M_I_SCALE*((AxzNode[3] - AxzNode[4] + AxzNode[7] - AxzNode[ 8] + AxzNode[11] - AxzNode[12] + AxzNode[14] - AxzNode[13] + AxzNode[17] - AxzNode[18]));
    Axz_qz_t30 = F_M_I_SCALE*((AxzNode[5] - AxzNode[6] + AxzNode[9] - AxzNode[10] + AxzNode[11] - AxzNode[12] + AxzNode[16] - AxzNode[15] + AxzNode[18] - AxzNode[17]));
}

Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);