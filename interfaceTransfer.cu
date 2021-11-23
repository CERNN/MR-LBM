#include "interfaceTransfer.cuh"


__device__ void gpuInterfacePushOffset(
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
    }   

    if (threadIdx.z == 0){ //b                                                                                                                                                                                     
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


__device__ void gpuInterfacePushCentered(
    dim3 threadIdx, dim3 blockIdx, dfloat pop[Q],
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1){


    // global x/y/z  
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    //thread x/y/z
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    
    //block xyz
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    if (threadIdx.x == 0) { //w
        fGhostX_0[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2]; 
        fGhostX_0[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
        fGhostX_0[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
        fGhostX_0[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
        fGhostX_0[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
        #ifdef D3Q27                                                                                                           
        fGhostX_0[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
        fGhostX_0[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
        fGhostX_0[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
        fGhostX_0[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.x == (BLOCK_NX - 1)){                                                                                                                                                                               
        fGhostX_1[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
        fGhostX_1[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
        fGhostX_1[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
        fGhostX_1[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
        fGhostX_1[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
        #ifdef D3Q27                                                                                                           
        fGhostX_1[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
        fGhostX_1[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
        fGhostX_1[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
        fGhostX_1[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27       
    }
    if (threadIdx.y == 0)  { //s                                                                                                                                                                                        
        fGhostY_0[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 4];
        fGhostY_0[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 8];
        fGhostY_0[idxPopY(tx, tz, 2, bx, by, bz)] = pop[12];
        fGhostY_0[idxPopY(tx, tz, 3, bx, by, bz)] = pop[13];
        fGhostY_0[idxPopY(tx, tz, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        fGhostY_0[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
        fGhostY_0[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
        fGhostY_0[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
        fGhostY_0[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.y == (BLOCK_NY - 1)){                                                                                                                                                                        
        fGhostY_1[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 3];
        fGhostY_1[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 7];
        fGhostY_1[idxPopY(tx, tz, 2, bx, by, bz)] = pop[11];
        fGhostY_1[idxPopY(tx, tz, 3, bx, by, bz)] = pop[14];
        fGhostY_1[idxPopY(tx, tz, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        fGhostY_1[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
        fGhostY_1[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
        fGhostY_1[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
        fGhostY_1[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }
    if (threadIdx.z == 0){ //b                                                                                                                                                                                     
        fGhostZ_0[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 6];
        fGhostZ_0[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[10];
        fGhostZ_0[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[12];
        fGhostZ_0[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[15];
        fGhostZ_0[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        fGhostZ_0[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
        fGhostZ_0[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
        fGhostZ_0[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
        fGhostZ_0[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.z == (BLOCK_NZ - 1)){                                                                                                               
        fGhostZ_1[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 5];
        fGhostZ_1[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[ 9];
        fGhostZ_1[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[11];
        fGhostZ_1[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[16];
        fGhostZ_1[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        fGhostZ_1[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
        fGhostZ_1[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
        fGhostZ_1[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
        fGhostZ_1[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                                                                                                                                    
    }

}


__device__ void gpuInterfacePullOffset(
    dim3 threadIdx, dim3 blockIdx, dfloat* pop,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1){

    // global x/y/z  
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    //thread x/y/z
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;


    /* ------------------------------ CORNER ------------------------------ */
    if(      ty == 0            && tx == 0              && tz == 0)             {//swb
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];    
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
//
        pop[ 7] = fGhostY_1[idxPopY((tx-1+BLOCK_NX)%BLOCK_NX, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; //y
        pop[ 9] = fGhostX_1[idxPopX(ty, (tz-1+BLOCK_NZ)%BLOCK_NZ, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, (bz-1-NUM_BLOCK_Z)%NUM_BLOCK_Z)]; //z
        pop[11] = fGhostZ_1[idxPopZ(tx, (ty-1+BLOCK_NY)%BLOCK_NY, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)]; //z
//
        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[15] = fGhostX_1[idxPopX(ty, tz+1, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[17] = fGhostY_1[idxPopY(tx, tz+1, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[18] = fGhostZ_1[idxPopZ(tx, ty+1, 4, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];     
    }else if(ty == 0            && tx == 0              && tz == (BLOCK_NZ-1))  {//swf
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];    
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
//
        pop[ 7] = fGhostX_1[idxPopX((ty-1+BLOCK_NY)%BLOCK_NY, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; //y
        pop[15] = fGhostX_1[idxPopX(ty, (tz+1)%BLOCK_NZ, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, (bz+1)%NUM_BLOCK_Z)]; //z
        pop[17] = fGhostY_1[idxPopY(tx, (tz+1)%BLOCK_NZ, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, (bz+1)%NUM_BLOCK_Z)]; //z
//
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
    }else if(ty == 0            && tx == (BLOCK_NX-1)   && tz == 0)             {//seb
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];    
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
//
        pop[11] = fGhostY_1[idxPopY(tx, (tz-1+BLOCK_NZ)%BLOCK_NZ, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[14] = fGhostX_0[idxPopX((ty-1+BLOCK_NY)%BLOCK_NY, tz, 3, (bx+1)%NUM_BLOCK_X, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[16] = fGhostX_0[idxPopX(ty, (tz-1+BLOCK_NZ)%BLOCK_NZ, 4, (bx+1)%NUM_BLOCK_X, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
//
        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[ 9] = fGhostZ_1[idxPopZ(tx-1, ty, 1, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[10] = fGhostX_0[idxPopX(ty, tz+1, 2, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[17] = fGhostY_1[idxPopY(tx, tz+1, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[18] = fGhostZ_1[idxPopZ(tx, ty+1, 4, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)]; 
    }else if(ty == 0            && tx == (BLOCK_NX-1)   && tz == (BLOCK_NZ-1))  {//sef
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];  
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
//
        pop[10] = fGhostX_0[idxPopX(ty, (tz+1)%BLOCK_NZ, 2, (bx+1)%NUM_BLOCK_X, by, (bz+1)%NUM_BLOCK_Z)];
        pop[14] = fGhostX_0[idxPopX((ty-1+BLOCK_NY)%BLOCK_NY, tz, 3, (bx+1)%NUM_BLOCK_X, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[17] = fGhostY_1[idxPopY(tx, (tz+1)%BLOCK_NZ, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, (bz+1)%NUM_BLOCK_Z)];
//
        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[16] = fGhostX_0[idxPopX(ty, tz-1, 4, (bx+1)%NUM_BLOCK_X, by, bz)];
    }else if(ty == (BLOCK_NY-1) && tx == 0              && tz == 0)             {//nwb
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
//
        pop[ 9] = fGhostX_1[idxPopX(ty, (tz-1+BLOCK_NZ)%BLOCK_NZ, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[13] = fGhostX_1[idxPopX((ty+1)%BLOCK_NY, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, (by+1)%NUM_BLOCK_Y, bz)];
        pop[18] = fGhostY_0[idxPopY(tx, (tz-1+BLOCK_NZ)%BLOCK_NZ, 4, bx, (by+1)%NUM_BLOCK_Y, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
//0         
        pop[ 7] = fGhostX_1[idxPopX(ty-1, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 8] = fGhostY_0[idxPopY(tx+1, tz, 1, bx, (by+1)%NUM_BLOCK_Y, bz)];
        pop[11] = fGhostZ_1[idxPopZ(tx, ty-1, 2, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[12] = fGhostY_0[idxPopY(tx, tz+1, 2, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[15] = fGhostX_1[idxPopX(ty, tz+1, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
    }else if(ty == (BLOCK_NY-1) && tx == 0              && tz == (BLOCK_NZ-1))  {//nwf
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];

        pop[12] = fGhostZ_0[idxPopZ(tx, (ty+1)%BLOCK_NY, 2, bx, (by+1)%NUM_BLOCK_Y, (bz+1)%NUM_BLOCK_Z)];
        pop[13] = fGhostX_1[idxPopX((ty+1)%BLOCK_NY, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, (by+1)%NUM_BLOCK_Y, bz)];
        pop[15] = fGhostX_1[idxPopX(ty, (tz+1)%BLOCK_NZ, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, (bz+1)%NUM_BLOCK_Z)];

        pop[ 7] = fGhostX_1[idxPopX(ty-1, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 8] = fGhostY_0[idxPopY(tx+1, tz, 1, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[18] = fGhostY_0[idxPopY(tx, tz-1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
    }else if(ty == (BLOCK_NY-1) && tx == (BLOCK_NX-1)   && tz == 0)             {//neb
        pop[ 2] =  fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 4] =  fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 5] =  fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        
        pop[ 8] =  fGhostX_0[idxPopX((ty+1)%BLOCK_NY, tz, 1, (bx+1)%NUM_BLOCK_X, (by+1)%NUM_BLOCK_Y, bz)]; //y
        pop[16] =  fGhostZ_1[idxPopZ((tx+1)%BLOCK_NX, ty, 3, (bx+1)%NUM_BLOCK_X, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[18] =  fGhostZ_1[idxPopZ(tx, (ty+1)%BLOCK_NY, 4, bx, (by+1)%NUM_BLOCK_Y, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];  
// 
        pop[ 9] =  fGhostZ_1[idxPopZ(tx-1, ty, 1, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[10] =  fGhostX_0[idxPopX(ty, tz+1, 2, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[11] =  fGhostZ_1[idxPopZ(tx, ty-1, 2, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[12] =  fGhostY_0[idxPopY(tx, tz+1, 2, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[13] =  fGhostY_0[idxPopY(tx+1, tz, 3, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[14] =  fGhostX_0[idxPopX(ty-1, tz, 3, (bx+1)%NUM_BLOCK_X, by, bz)];
    }else if(ty == (BLOCK_NY-1) && tx == (BLOCK_NX-1)   && tz == (BLOCK_NZ-1))  {//nef
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];

        pop[ 8] = fGhostY_0[idxPopY((tx+1)%BLOCK_NX, tz, 1, (bx+1)%NUM_BLOCK_X, (by+1)%NUM_BLOCK_Y, bz)];  //x
        pop[10] = fGhostZ_0[idxPopZ((tx+1)%BLOCK_NX, ty, 1, (bx+1)%NUM_BLOCK_X, by, (bz+1)%NUM_BLOCK_Z)];
        pop[12] = fGhostZ_0[idxPopZ(tx, (ty+1)%BLOCK_NY, 2, bx, (by+1)%NUM_BLOCK_X, (bz+1)%NUM_BLOCK_Z)];   //y
//
        pop[13] = fGhostY_0[idxPopY(tx-1, tz, 3, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[14] = fGhostX_0[idxPopX(ty-1, tz, 3, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[16] = fGhostX_0[idxPopX(ty, tz-1, 4, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[18] = fGhostY_0[idxPopY(tx, tz-1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)]; 

/* ------------------------------ EDGE ------------------------------ */
    }else if(ty == 0            && tx == 0)                     {//sw
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];  
        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; //
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[15] = fGhostX_1[idxPopX(ty, tz+1, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[17] = fGhostY_1[idxPopY(tx, tz+1, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];        

        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; 
    }else if(ty == 0            && tx == (BLOCK_NX-1))       {//se
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; 
        pop[10] = fGhostX_0[idxPopX(ty, tz+1, 2, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; //
        pop[16] = fGhostX_0[idxPopX(ty, tz-1, 4, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[17] = fGhostY_1[idxPopY(tx, tz+1, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; 

        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)]; 
    }else if(ty == (BLOCK_NY-1) && tx == 0)                  {//nw
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[12] = fGhostY_0[idxPopY(tx, tz+1, 2, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)]; //
        pop[15] = fGhostX_1[idxPopX(ty, tz+1, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[18] = fGhostY_0[idxPopY(tx, tz-1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)];  

        pop[ 7] = fGhostX_1[idxPopX(ty-1, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 8] = fGhostY_0[idxPopY(tx+1, tz, 1, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
    }else if(ty == (BLOCK_NY-1) && tx == (BLOCK_NX-1))    {//ne
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)]; //
        pop[10] = fGhostX_0[idxPopX(ty, tz+1, 2, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[12] = fGhostY_0[idxPopY(tx, tz+1, 2, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[16] = fGhostX_0[idxPopX(ty, tz-1, 4, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[18] = fGhostY_0[idxPopY(tx, tz-1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
         
        pop[13] = fGhostY_0[idxPopY(tx-1, tz, 3, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[14] = fGhostX_0[idxPopX(ty-1, tz, 3, (bx+1)%NUM_BLOCK_X, by, bz)]; 
    }else if(ty == 0            && tz == 0)                     {//sb
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)]; 
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[ 9] = fGhostZ_1[idxPopZ(tx-1, ty, 1, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];//
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];

        pop[17] = fGhostY_1[idxPopY(tx, tz+1, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[18] = fGhostZ_1[idxPopZ(tx, ty+1, 4, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
    }else if(ty == 0            && tz == (BLOCK_NZ-1))       {//sf
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];    
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)]; //

        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)]; 
    }else if(ty == (BLOCK_NY-1) && tz == 0)                  {//nb
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[ 8] = fGhostY_0[idxPopY(tx+1, tz, 1, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 9] = fGhostZ_1[idxPopZ(tx-1, ty, 1, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[13] = fGhostY_0[idxPopY(tx-1, tz, 3, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[18] = fGhostY_0[idxPopY(tx, tz-1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)]; //

        pop[11] = fGhostZ_1[idxPopZ(tx, ty-1, 2, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[12] = fGhostY_0[idxPopY(tx, tz+1, 2, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
    }else if(ty == (BLOCK_NY-1) && tz == (BLOCK_NZ-1))    {//nf
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[ 8] = fGhostY_0[idxPopY(tx+1, tz, 1, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)]; //<<
        pop[13] = fGhostY_0[idxPopY(tx+1, tz, 3, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];
        //0; //
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[18] = fGhostY_0[idxPopY(tx, tz+1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
    }else if(tx == 0            && tz == 0)                     {//wb
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[ 7] = fGhostX_1[idxPopX(ty-1, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];//
        pop[11] = fGhostZ_1[idxPopZ(tx, ty-1, 2, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[18] = fGhostZ_1[idxPopZ(tx, ty+1, 4, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];   
        //
        pop[15] = fGhostX_1[idxPopX(ty, tz+1, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
    }else if(tx == 0            && tz == (BLOCK_NZ-1))       {//wf
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[ 7] = fGhostX_1[idxPopX(ty-1, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];//
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)];
        //0;//
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];
    }else if(tx == (BLOCK_NX-1) && tz == 0)                  {//eb
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[11] = fGhostZ_1[idxPopZ(tx, ty-1, 2, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[14] = fGhostX_0[idxPopX(ty-1, tz, 3, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];//
        pop[18] = fGhostZ_1[idxPopZ(tx, ty+1, 4, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        //
        pop[ 9] = fGhostZ_1[idxPopZ(tx-1, ty, 1, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[10] = fGhostX_0[idxPopX(ty, tz+1, 2, (bx+1)%NUM_BLOCK_X, by, bz)];

    }else if(tx == (BLOCK_NX-1) && tz == (BLOCK_NZ-1))    {//ef
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];//
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[14] = fGhostX_0[idxPopX(ty-1, tz, 3, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)];
        //
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[16] = fGhostX_0[idxPopX(ty, tz-1, 4, (bx+1)%NUM_BLOCK_X, by, bz)];

/* ------------------------------ FACE ------------------------------ */
    }else if (tx == 0) { //w
        pop[ 1] = fGhostX_1[idxPopX(ty, tz, 0,   (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 7] = fGhostX_1[idxPopX(ty-1, tz, 1, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[ 9] = fGhostX_1[idxPopX(ty, tz-1, 2, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[13] = fGhostX_1[idxPopX(ty+1, tz, 3, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
        pop[15] = fGhostX_1[idxPopX(ty, tz+1, 4, (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X, by, bz)];
    }else if (tx == (BLOCK_NX - 1)){ //e 
        pop[ 2] = fGhostX_0[idxPopX(ty,   tz, 0, (bx+1)%NUM_BLOCK_X, by, bz)]; 
        pop[ 8] = fGhostX_0[idxPopX(ty+1, tz, 1, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[10] = fGhostX_0[idxPopX(ty, tz+1, 2, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[14] = fGhostX_0[idxPopX(ty-1, tz, 3, (bx+1)%NUM_BLOCK_X, by, bz)];
        pop[16] = fGhostX_0[idxPopX(ty, tz-1, 4, (bx+1)%NUM_BLOCK_X, by, bz)];


    }else if (ty == 0)  { //s  
        pop[ 3] = fGhostY_1[idxPopY(tx,   tz, 0, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];    
        pop[ 7] = fGhostY_1[idxPopY(tx-1, tz, 1, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[11] = fGhostY_1[idxPopY(tx, tz-1, 2, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[14] = fGhostY_1[idxPopY(tx+1, tz, 3, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
        pop[17] = fGhostY_1[idxPopY(tx, tz+1, 4, bx, (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y, bz)];
    }else if (ty == (BLOCK_NY - 1)){ //n  
        pop[ 4] = fGhostY_0[idxPopY(tx,   tz, 0, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[ 8] = fGhostY_0[idxPopY(tx+1, tz, 1, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[12] = fGhostY_0[idxPopY(tx, tz+1, 2, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[13] = fGhostY_0[idxPopY(tx-1, tz, 3, bx, (by+1)%NUM_BLOCK_Y, bz)]; 
        pop[18] = fGhostY_0[idxPopY(tx, tz-1, 4, bx, (by+1)%NUM_BLOCK_Y, bz)]; 


    }else if (tz == 0){ //b  
        pop[ 5] = fGhostZ_1[idxPopZ(tx,   ty, 0, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[ 9] = fGhostZ_1[idxPopZ(tx-1, ty, 1, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[11] = fGhostZ_1[idxPopZ(tx, ty-1, 2, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[16] = fGhostZ_1[idxPopZ(tx+1, ty, 3, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];
        pop[18] = fGhostZ_1[idxPopZ(tx, ty+1, 4, bx, by, (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)];                                                                                  
    }else if (tz == (BLOCK_NZ - 1)) { //f   
        pop[ 6] = fGhostZ_0[idxPopZ(tx,   ty, 0, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[10] = fGhostZ_0[idxPopZ(tx+1, ty, 1, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[12] = fGhostZ_0[idxPopZ(tx, ty+1, 2, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[15] = fGhostZ_0[idxPopZ(tx-1, ty, 3, bx, by, (bz+1)%NUM_BLOCK_Z)];
        pop[17] = fGhostZ_0[idxPopZ(tx, ty-1, 4, bx, by, (bz+1)%NUM_BLOCK_Z)];
    }
}
__device__ void gpuInterfacePullCentered(
    dim3 threadIdx, dim3 blockIdx, dfloat* pop,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1){
    if (threadIdx.x == 0)
    {
        pop[ 1] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 7] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 9] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[13] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[15] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[19] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[21] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[23] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[26] = fGhostX_0[idxPopX(threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    else if (threadIdx.x == BLOCK_NX - 1)
    {
        pop[ 2] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 8] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[10] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[14] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[16] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[20] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[22] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[24] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[25] = fGhostX_1[idxPopX(threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }   
    if (threadIdx.y == 0)
    {
        pop[ 3] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 7] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[11] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[14] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[17] = fGhostY_0[idxPopY(threadIdx.x, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[19] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[21] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[24] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[25] = fGhostY_0[idxPopX(threadIdx.x, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    else if (threadIdx.y == BLOCK_NY - 1)
    {
        pop[ 4] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 8] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[12] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[13] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[18] = fGhostY_1[idxPopY(threadIdx.x, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[20] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[22] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[23] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[26] = fGhostY_1[idxPopX(threadIdx.x, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    if (threadIdx.z == 0)
    {
        pop[ 5] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[ 9] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[11] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[16] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[18] = fGhostZ_0[idxPopZ(threadIdx.x, threadIdx.y, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[19] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[22] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[23] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[25] = fGhostZ_0[idxPopX(threadIdx.x, threadIdx.y, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
    else if (threadIdx.z == BLOCK_NZ - 1)
    {
        pop[ 6] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[10] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[12] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[15] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[17] = fGhostZ_1[idxPopZ(threadIdx.x, threadIdx.y, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
        #ifdef D3Q27
        pop[20] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[21] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[24] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
        pop[26] = fGhostZ_1[idxPopX(threadIdx.x, threadIdx.y, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
        #endif //D3Q27
    }
}