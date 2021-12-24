#include "interfaceTransfer.cuh"



   /*
    OBS.
    Write cost is higher than read cost, because of that gpuInterfacePush
    is coalesced, and gpuInterfacePull is not
    */


__device__ void gpuInterfacePush(
    dim3 threadIdx, dim3 blockIdx, dfloat pop[Q],
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1){

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


__device__ void gpuInterfacePull(
    dim3 threadIdx, dim3 blockIdx, dfloat* pop,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1){


    //thread x/y/z
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    if ((tx>0 && tx<BLOCK_NX-1)&&(ty>0 && ty<BLOCK_NY-1)&&(tz>0 && tz<BLOCK_NZ-1))
        return;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    int txp1 = (tx+1)%BLOCK_NX ;
    int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    int typ1 = (ty+1)%BLOCK_NY;
    int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    int tzp1 = (tz+1)%BLOCK_NZ ;

    int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    int bxp1 = (bx+1)%NUM_BLOCK_X;
    int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    int byp1 = (by+1)%NUM_BLOCK_Y;
    int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    int bzp1 = (bz+1)%NUM_BLOCK_Z;


    if (tx == 0) { //w
        pop[ 1] = fGhostX_1[idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
        pop[ 7] = fGhostX_1[idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
        pop[ 9] = fGhostX_1[idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
        pop[13] = fGhostX_1[idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        pop[15] = fGhostX_1[idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #ifdef D3Q27
        pop[19] = fGhostX_1[idxPopX(tym1, tzm1, 5, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
        pop[21] = fGhostX_1[idxPopX(tym1, tzp1, 6, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[23] = fGhostX_1[idxPopX(typ1, tzm1, 7, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
        pop[26] = fGhostX_1[idxPopX(typ1, tzp1, 8, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #endif //D3Q27
    }
    else if (tx == (BLOCK_NX - 1))
    { //e
        pop[ 2] = fGhostX_0[idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
        pop[ 8] = fGhostX_0[idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        pop[10] = fGhostX_0[idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[14] = fGhostX_0[idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
        pop[16] = fGhostX_0[idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
        #ifdef D3Q27
        pop[20] = fGhostX_0[idxPopX(typ1, tzp1, 5, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[22] = fGhostX_0[idxPopX(typ1, tzm1, 6, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
        pop[24] = fGhostX_0[idxPopX(tym1, tzp1, 7, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[25] = fGhostX_0[idxPopX(tym1, tzm1, 8, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
        #endif //D3Q27
    }

    if (ty == 0)
    { //s
        pop[ 3] = fGhostY_1[idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
        pop[ 7] = fGhostY_1[idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
        pop[11] = fGhostY_1[idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
        pop[14] = fGhostY_1[idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
        pop[17] = fGhostY_1[idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #ifdef D3Q27
        pop[19] = fGhostY_1[idxPopY(txm1, tzm1, 5, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == 0) ? bzm1 : bz))];
        pop[21] = fGhostY_1[idxPopY(txm1, tzp1, 6, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[24] = fGhostY_1[idxPopY(txp1, tzp1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[25] = fGhostY_1[idxPopY(txp1, tzm1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == 0) ? bzm1 : bz))];
        #endif //D3Q27
    }
    else if (ty == (BLOCK_NY - 1))
    { //n
        pop[ 4] = fGhostY_0[idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
        pop[ 8] = fGhostY_0[idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
        pop[12] = fGhostY_0[idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[13] = fGhostY_0[idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
        pop[18] = fGhostY_0[idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
        #ifdef D3Q27
        pop[20] = fGhostY_0[idxPopY(txp1, tzp1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[22] = fGhostY_0[idxPopY(txp1, tzm1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == 0) ? bzm1 : bz))];
        pop[23] = fGhostY_0[idxPopY(txm1, tzm1, 7, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == 0) ? bzm1 : bz))];
        pop[26] = fGhostY_0[idxPopY(txm1, tzp1, 8, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #endif //D3Q27
    }

    if (tz == 0)
    { //b
        pop[ 5] = fGhostZ_1[idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
        pop[ 9] = fGhostZ_1[idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
        pop[11] = fGhostZ_1[idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
        pop[16] = fGhostZ_1[idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
        pop[18] = fGhostZ_1[idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        #ifdef D3Q27
        pop[19] = fGhostZ_1[idxPopZ(txm1, tym1, 5, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzm1)];
        pop[22] = fGhostZ_1[idxPopZ(txp1, typ1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        pop[23] = fGhostZ_1[idxPopZ(txm1, typ1, 7, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        pop[25] = fGhostZ_1[idxPopZ(txp1, tym1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzm1)];
        #endif //D3Q27
    }
    else if (tz == (BLOCK_NZ - 1))
    { //f
        pop[ 6] = fGhostZ_0[idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
        pop[10] = fGhostZ_0[idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
        pop[12] = fGhostZ_0[idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        pop[15] = fGhostZ_0[idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
        pop[17] = fGhostZ_0[idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
        #ifdef D3Q27
        pop[20] = fGhostZ_0[idxPopZ(txp1, typ1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        pop[21] = fGhostZ_0[idxPopZ(txm1, tym1, 6, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzp1)];
        pop[24] = fGhostZ_0[idxPopZ(txp1, tym1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzp1)];
        pop[26] = fGhostZ_0[idxPopZ(txm1, typ1, 8, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        #endif //D3Q27
    }
}
