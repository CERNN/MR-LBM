dfloat AyzNode[GQ];

dfloat invAyz = 1/AyzVar;
dfloat Ayz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Ayz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Ayz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

//dfloat udx_t30 = 0.0;
//dfloat udy_t30 = 0.0;
//dfloat udz_t30 = 0.0;

#include COLREC_AYZ_RECONSTRUCTION

__syncthreads();

//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = AyzNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = AyzNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = AyzNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = AyzNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = AyzNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = AyzNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = AyzNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = AyzNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = AyzNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = AyzNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = AyzNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = AyzNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = AyzNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = AyzNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = AyzNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = AyzNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = AyzNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = AyzNode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
AyzNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
AyzNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
AyzNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
AyzNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
AyzNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
AyzNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
AyzNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
AyzNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
AyzNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
AyzNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
AyzNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
AyzNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
AyzNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
AyzNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
AyzNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
AyzNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
AyzNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
AyzNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Ayz.inc"

if(nodeType != BULK){
    #include CASE_AYZ_BC_DEF
}else{
    AyzVar = AyzNode[0] + AyzNode[1] + AyzNode[2] + AyzNode[3] + AyzNode[4] + AyzNode[5] + AyzNode[6] + AyzNode[7] + AyzNode[8] + AyzNode[9] + AyzNode[10] + AyzNode[11] + AyzNode[12] + AyzNode[13] + AyzNode[14] + AyzNode[15] + AyzNode[16] + AyzNode[17] + AyzNode[18];
    AyzVar = AyzVar; // + T_Q_INTERNAL_D_Cp;
    invAyz= 1.0/AyzVar;

    Ayz_qx_t30 = F_M_I_SCALE*((AyzNode[1] - AyzNode[2] + AyzNode[7] - AyzNode[ 8] + AyzNode[ 9] - AyzNode[10] + AyzNode[13] - AyzNode[14] + AyzNode[15] - AyzNode[16]));
    Ayz_qy_t30 = F_M_I_SCALE*((AyzNode[3] - AyzNode[4] + AyzNode[7] - AyzNode[ 8] + AyzNode[11] - AyzNode[12] + AyzNode[14] - AyzNode[13] + AyzNode[17] - AyzNode[18]));
    Ayz_qz_t30 = F_M_I_SCALE*((AyzNode[5] - AyzNode[6] + AyzNode[9] - AyzNode[10] + AyzNode[11] - AyzNode[12] + AyzNode[16] - AyzNode[15] + AyzNode[18] - AyzNode[17]));
}

