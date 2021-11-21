#include "interfaceSpread.cuh"

__device__ void gpuInterfaceSpread(
    dim3 threadIdx, dim3 blockIdx, dfloat pop[Q],
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1){


    // global x/y/z  
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
  
    int xp1 = (x + 1 + NX) % NX;
    int xm1 = (x - 1 + NX) % NX;

    int yp1 = (y + 1 + NY) % NY;
    int ym1 = (y - 1 + NY) % NY;

    int zp1 = (z + 1 + NZ_TOTAL) % NZ_TOTAL;
    int zm1 = (z - 1 + NZ_TOTAL) % NZ_TOTAL;

    //thread x/y/z
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int txp1 = (xp1+BLOCK_NX)%BLOCK_NX;
    int txm1 = (xm1+BLOCK_NX)%BLOCK_NX;

    int typ1 = (yp1+BLOCK_NY)%BLOCK_NY;
    int tym1 = (ym1+BLOCK_NY)%BLOCK_NY;

    int tzp1 = (zp1+BLOCK_NZ)%BLOCK_NZ;
    int tzm1 = (zm1+BLOCK_NZ)%BLOCK_NZ;
    
    //block xyz
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int bxp1 = xp1/(int)BLOCK_NX;
    int bxm1 = xm1/(int)BLOCK_NX;    

    int byp1 = yp1/(int)BLOCK_NY;
    int bym1 = ym1/(int)BLOCK_NY;    

    int bzp1 = zp1/(int)BLOCK_NZ;
    int bzm1 = zm1/(int)BLOCK_NZ;

    if (threadIdx.y == 0)  { //s                                                                                                                                                                                        
        fGhostY_1[idxPopY(tx, tz, 0, bx, bym1, bz)] = pop[ 4];
        fGhostY_1[idxPopY(txm1, tz, 1, bxm1, bym1, bz)] = pop[ 8];
        fGhostY_1[idxPopY(tx, tzm1, 2, bx, bym1, bzm1)] = pop[12];
        fGhostY_1[idxPopY(txp1, tz, 3, bxp1, bym1, bz)] = pop[13];
        fGhostY_1[idxPopY(tx, tzp1, 4, bx, bym1, bzp1)] = pop[18];
        #ifdef D3Q27                                                                                                           
        fGhostY_1[idxPopY(txm1, tzm1, 5, bxm1, bym1, bzm1)] = pop[20];
        fGhostY_1[idxPopY(txm1, tzp1, 6, bxm1, bym1, bzp1)] = pop[22];
        fGhostY_1[idxPopY(txp1, tzp1, 7, bxp1, bym1, bzp1)] = pop[23];
        fGhostY_1[idxPopY(txp1, tzm1, 8, bxp1, bym1, bzm1)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.y == (BLOCK_NY - 1)){                                                                                                                                                                        
        fGhostY_0[idxPopY(tx, tz, 0, bx, byp1, bz)] = pop[ 3];
        fGhostY_0[idxPopY(txp1, tz, 1, bxp1, byp1, bz)] = pop[ 7];
        fGhostY_0[idxPopY(tx, tzp1, 2, bx, byp1, bzp1)] = pop[11];
        fGhostY_0[idxPopY(txm1, tz, 3, bxm1, byp1, bz)] = pop[14];
        fGhostY_0[idxPopY(tx, tzm1, 4, bx, byp1, bzm1)] = pop[17];
        #ifdef D3Q27                                                                                                           
        fGhostY_0[idxPopY(txp1, tzp1, 5, bxp1, byp1, bzp1)] = pop[19];
        fGhostY_0[idxPopY(txp1, tzm1, 6, bxp1, byp1, bzm1)] = pop[21];
        fGhostY_0[idxPopY(txm1, tzm1, 7, bxm1, byp1, bzm1)] = pop[24];
        fGhostY_0[idxPopY(txm1, tzp1, 8, bxm1, byp1, bzp1)] = pop[25];
        #endif //D3Q27                                                                                                           
    }
    
    if (threadIdx.x == 0) { //w
        fGhostX_1[idxPopX(ty, tz, 0, bxm1, by, bz)] = pop[ 2]; 
        fGhostX_1[idxPopX(tym1, tz, 1, bxm1, bym1, bz)] = pop[ 8];
        fGhostX_1[idxPopX(ty, tzm1, 2, bxm1, by, bzm1)] = pop[10];
        fGhostX_1[idxPopX(typ1, tz, 3, bxm1, byp1, bz)] = pop[14];
        fGhostX_1[idxPopX(ty, tzp1, 4, bxm1, by, bzp1)] = pop[16];
        #ifdef D3Q27                                                                                                           
        fGhostX_1[idxPopX(tym1, tzm1, 5, bxm1, bym1, bzm1)] = pop[20];
        fGhostX_1[idxPopX(tym1, tzp1, 6, bxm1, bym1, bzp1)] = pop[22];
        fGhostX_1[idxPopX(typ1, tzm1, 7, bxm1, byp1, bzm1)] = pop[24];
        fGhostX_1[idxPopX(typ1, tzp1, 8, bxm1, byp1, bzp1)] = pop[25];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.x == (BLOCK_NX - 1)){                                                                                                                                                                               
        fGhostX_0[idxPopX(ty, tz, 0, bxp1, by, bz)] = pop[ 1];
        fGhostX_0[idxPopX(typ1, tz, 1, bxp1, byp1, bz)] = pop[ 7];
        fGhostX_0[idxPopX(ty, tzp1, 2, bxp1, by, bzp1)] = pop[ 9];
        fGhostX_0[idxPopX(tym1, tz, 3, bxp1, bym1, bz)] = pop[13];
        fGhostX_0[idxPopX(ty, tzm1, 4, bxp1, by, bzm1)] = pop[15];
        #ifdef D3Q27                                                                                                           
        fGhostX_0[idxPopX(typ1, tzp1, 5, bxp1, byp1, bzp1)] = pop[19];
        fGhostX_0[idxPopX(typ1, tzm1, 6, bxp1, byp1, bzm1)] = pop[21];
        fGhostX_0[idxPopX(tym1, tzp1, 7, bxp1, bym1, bzp1)] = pop[23];
        fGhostX_0[idxPopX(tym1, tzm1, 8, bxp1, bym1, bzm1)] = pop[26];
        #endif //D3Q27       

    }if (threadIdx.z == 0){ //b                                                                                                                                                                                     
        fGhostZ_1[idxPopZ(tx, ty, 0, bx, by, bzm1)] = pop[ 6];
        fGhostZ_1[idxPopZ(txm1, ty, 1, bxm1, by, bzm1)] = pop[10];
        fGhostZ_1[idxPopZ(tx, tym1, 2, bx, bym1, bzm1)] = pop[12];
        fGhostZ_1[idxPopZ(txp1, ty, 3, bxp1, by, bzm1)] = pop[15];
        fGhostZ_1[idxPopZ(tx, typ1, 4, bx, byp1, bzm1)] = pop[17];
        #ifdef D3Q27                                                                                                           
        fGhostZ_1[idxPopZ(txm1, tym1, 5, bxm1, bym1, bzm1)] = pop[20];
        fGhostZ_1[idxPopZ(txp1, typ1, 6, bxp1, byp1, bzm1)] = pop[21];
        fGhostZ_1[idxPopZ(txm1, typ1, 7, bxm1, byp1, bzm1)] = pop[24];
        fGhostZ_1[idxPopZ(txp1, tym1, 8, bxp1, bym1, bzm1)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.z == (BLOCK_NZ - 1)){                                                                                                               
        fGhostZ_0[idxPopZ(tx, ty, 0, bx, by, bzp1)] = pop[ 5];
        fGhostZ_0[idxPopZ(txp1, ty, 1, bxp1, by, bzp1)] = pop[ 9];
        fGhostZ_0[idxPopZ(tx, typ1, 2, bx, byp1, bzp1)] = pop[11];
        fGhostZ_0[idxPopZ(txm1, ty, 3, bxm1, by, bzp1)] = pop[16];
        fGhostZ_0[idxPopZ(tx, tym1, 4, bx, bym1, bzp1)] = pop[18];
        #ifdef D3Q27                                                                                                           
        fGhostZ_0[idxPopZ(txp1, typ1, 5, bxp1, byp1, bzp1)] = pop[19];
        fGhostZ_0[idxPopZ(txm1, tym1, 6, bxm1, bym1, bzp1)] = pop[22];
        fGhostZ_0[idxPopZ(txp1, tym1, 7, bxp1, bym1, bzp1)] = pop[23];
        fGhostZ_0[idxPopZ(txm1, typ1, 8, bxm1, byp1, bzp1)] = pop[25];
        #endif //D3Q27                                                                                                                                                                                                                    
    }

    }
