dfloat AzzNode[GQ];

dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat invAzz = 1/AzzVar;
dfloat Azz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Azz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
dfloat Azz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

//dfloat udx_t30 = 0.0;
//dfloat udy_t30 = 0.0;
//dfloat udz_t30 = 0.0;

#include COLREC_AZZ_RECONSTRUCTION

__syncthreads();

//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = AzzNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = AzzNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = AzzNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = AzzNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = AzzNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = AzzNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = AzzNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = AzzNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = AzzNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = AzzNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = AzzNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = AzzNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = AzzNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = AzzNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = AzzNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = AzzNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = AzzNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = AzzNode[18];

//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
AzzNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
AzzNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
AzzNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
AzzNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
AzzNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
AzzNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
AzzNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
AzzNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
AzzNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
AzzNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
AzzNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
AzzNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
AzzNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
AzzNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
AzzNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
AzzNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
AzzNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
AzzNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

/* load pop from global in cover nodes */

#include "includeFiles/conformationTransport/popLoad_Azz.inc"

if(nodeType != BULK){
    #include CASE_AZZ_BC_DEF
}else{
    AzzVar = AzzNode[0] + AzzNode[1] + AzzNode[2] + AzzNode[3] + AzzNode[4] + AzzNode[5] + AzzNode[6] + AzzNode[7] + AzzNode[8] + AzzNode[9] + AzzNode[10] + AzzNode[11] + AzzNode[12] + AzzNode[13] + AzzNode[14] + AzzNode[15] + AzzNode[16] + AzzNode[17] + AzzNode[18];
    AzzVar = AzzVar; // + T_Q_INTERNAL_D_Cp;
    invAzz= 1.0/AzzVar;

    Azz_qx_t30 = F_M_I_SCALE*((AzzNode[1] - AzzNode[2] + AzzNode[7] - AzzNode[ 8] + AzzNode[ 9] - AzzNode[10] + AzzNode[13] - AzzNode[14] + AzzNode[15] - AzzNode[16]));
    Azz_qy_t30 = F_M_I_SCALE*((AzzNode[3] - AzzNode[4] + AzzNode[7] - AzzNode[ 8] + AzzNode[11] - AzzNode[12] + AzzNode[14] - AzzNode[13] + AzzNode[17] - AzzNode[18]));
    Azz_qz_t30 = F_M_I_SCALE*((AzzNode[5] - AzzNode[6] + AzzNode[9] - AzzNode[10] + AzzNode[11] - AzzNode[12] + AzzNode[16] - AzzNode[15] + AzzNode[18] - AzzNode[17]));
}

