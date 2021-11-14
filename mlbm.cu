#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(
    Moments mom,
    Populations pop)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;



    size_t indexNodeLBM = idxScalarGlobal(x,y,z);
    size_t indexBlock = idxBlock(blockIdx.x,blockIdx.y,blockIdx.z);

    dfloat rhoVar = mom.rho[indexNodeLBM];

    dfloat uxVar = mom.ux[indexNodeLBM];
    dfloat uyVar = mom.uy[indexNodeLBM];
    dfloat uzVar = mom.uz[indexNodeLBM];

    dfloat pxxVar = mom.pxx[indexNodeLBM];
    dfloat pxyVar = mom.pxy[indexNodeLBM];
    dfloat pxzVar = mom.pxz[indexNodeLBM];
    dfloat pyyVar = mom.pyy[indexNodeLBM];
    dfloat pyzVar = mom.pyz[indexNodeLBM];
    dfloat pzzVar = mom.pzz[indexNodeLBM];

    dfloat fNodeEq[Q];
    dfloat fNodeNeq[Q];
    dfloat fPop[Q];
    dfloat fStream[Q];

    // CALCULATE EQUILIBRIUM

    // Moments

    dfloat pxx_eq = rhoVar * (uxVar * uxVar + cs2);
    dfloat pxy_eq = rhoVar * (uxVar * uyVar);
    dfloat pxz_eq = rhoVar * (uxVar * uzVar);
    dfloat pyy_eq = rhoVar * (uyVar * uyVar + cs2);
    dfloat pyz_eq = rhoVar * (uyVar * uzVar);
    dfloat pzz_eq = rhoVar * (uzVar * uzVar + cs2);

    // Calculate temporary variables
    const dfloat p1_muu15 = 1 - 1.5 * (uxVar * uxVar + uyVar * uyVar + uzVar * uzVar);
    const dfloat rhoW0 = rhoVar * W0;
    const dfloat rhoW1 = rhoVar * W1;
    const dfloat rhoW2 = rhoVar * W2;
    const dfloat W1t3d2 = W1 * 3.0 / 2.0;
    const dfloat W2t3d2 = W2 * 3.0 / 2.0;
    const dfloat W1t9d2 = W1t3d2 * 3.0;
    const dfloat W2t9d2 = W2t3d2 * 3.0;

    #ifdef D3Q27
    const dfloat rhoW3 = rhoVar * W3;
    const dfloat W3t9d2 = W3 * 9 / 2;
    #endif
    const dfloat ux3 = 3 * uxVar;
    const dfloat uy3 = 3 * uyVar;
    const dfloat uz3 = 3 * uzVar;

    // Calculate equilibrium fNodeEq
    fNodeEq[0 ] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNodeEq[1 ] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNodeEq[2 ] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNodeEq[3 ] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNodeEq[4 ] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNodeEq[5 ] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNodeEq[6 ] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNodeEq[7 ] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNodeEq[8 ] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNodeEq[9 ] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
    fNodeEq[10] = gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15);
    fNodeEq[11] = gpu_f_eq(rhoW2, uy3 + uz3, p1_muu15);
    fNodeEq[12] = gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15);
    fNodeEq[13] = gpu_f_eq(rhoW2, ux3 - uy3, p1_muu15);
    fNodeEq[14] = gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15);
    fNodeEq[15] = gpu_f_eq(rhoW2, ux3 - uz3, p1_muu15);
    fNodeEq[16] = gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15);
    fNodeEq[17] = gpu_f_eq(rhoW2, uy3 - uz3, p1_muu15);
    fNodeEq[18] = gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15);
    #ifdef D3Q27
    fNodeEq[19] = gpu_f_eq(rhoW3, ux3 + uy3 + uz3, p1_muu15);
    fNodeEq[20] = gpu_f_eq(rhoW3, -ux3 - uy3 - uz3, p1_muu15);
    fNodeEq[21] = gpu_f_eq(rhoW3, ux3 + uy3 - uz3, p1_muu15);
    fNodeEq[22] = gpu_f_eq(rhoW3, -ux3 - uy3 + uz3, p1_muu15);
    fNodeEq[23] = gpu_f_eq(rhoW3, ux3 - uy3 + uz3, p1_muu15);
    fNodeEq[24] = gpu_f_eq(rhoW3, -ux3 + uy3 - uz3, p1_muu15);
    fNodeEq[25] = gpu_f_eq(rhoW3, -ux3 + uy3 + uz3, p1_muu15);
    fNodeEq[26] = gpu_f_eq(rhoW3, ux3 - uy3 - uz3, p1_muu15);
    #endif

    // CALCULATE NON-EQUILIBRIUM POPULATIONS
    #pragma unroll
    for (int i = 0; i < Q; i++){
        fNodeNeq[i] = rhoVar * 1.5 * w[i] * 
                    (((cx[i] * cx[i] - cs2) * (pxxVar - pxx_eq) + //Q-iab*(m_ab - m_ab^eq)
                        2 * (cx[i] * cy[i]) * (pxyVar - pxy_eq) + 
                        2 * (cx[i] * cz[i]) * (pxzVar - pxz_eq) + 
                        (cy[i] * cy[i] - cs2) * (pyyVar - pyy_eq) + 
                        2 * (cy[i] * cz[i]) * (pyzVar - pyz_eq) + 
                        (cz[i] * cz[i] - cs2) * (pzzVar - pzz_eq)) -
                        cs2*(cx[i] * FX + cy[i] * FY + cz[i] * FZ)); //force term
    }

    //CALCULATE COLLISION POPULATIONS
    fPop[0 ] = fNodeEq[0 ] + fNodeNeq[0 ];
    fPop[1 ] = fNodeEq[1 ] + fNodeNeq[1 ];
    fPop[2 ] = fNodeEq[2 ] + fNodeNeq[2 ];
    fPop[3 ] = fNodeEq[3 ] + fNodeNeq[3 ];
    fPop[4 ] = fNodeEq[4 ] + fNodeNeq[4 ];
    fPop[5 ] = fNodeEq[5 ] + fNodeNeq[5 ];
    fPop[6 ] = fNodeEq[6 ] + fNodeNeq[6 ];
    fPop[7 ] = fNodeEq[7 ] + fNodeNeq[7 ];
    fPop[8 ] = fNodeEq[8 ] + fNodeNeq[8 ];
    fPop[9 ] = fNodeEq[9 ] + fNodeNeq[9 ];
    fPop[10] = fNodeEq[10] + fNodeNeq[10];
    fPop[11] = fNodeEq[11] + fNodeNeq[11];
    fPop[12] = fNodeEq[12] + fNodeNeq[12];
    fPop[13] = fNodeEq[13] + fNodeNeq[13];
    fPop[14] = fNodeEq[14] + fNodeNeq[14];
    fPop[15] = fNodeEq[15] + fNodeNeq[15];
    fPop[16] = fNodeEq[16] + fNodeNeq[16];
    fPop[17] = fNodeEq[17] + fNodeNeq[17];
    fPop[18] = fNodeEq[18] + fNodeNeq[18];
    #ifdef D3Q27
    fPop[19] = fNodeEq[19] + fNodeNeq[19];
    fPop[20] = fNodeEq[20] + fNodeNeq[20];
    fPop[21] = fNodeEq[21] + fNodeNeq[21];
    fPop[22] = fNodeEq[22] + fNodeNeq[22];
    fPop[23] = fNodeEq[23] + fNodeNeq[23];
    fPop[24] = fNodeEq[24] + fNodeNeq[24];
    fPop[25] = fNodeEq[25] + fNodeNeq[25];
    fPop[26] = fNodeEq[26] + fNodeNeq[26];
    #endif


    __shared__ dfloat stream_population[BLOCK_LBM_SIZE*Q];

    //save populations in shared memory
    #pragma unroll
    for (int i = 0; i < Q; i++){
        stream_population[idxPopBlock(threadIdx.x,threadIdx.y,threadIdx.z,i)] = fPop[i];
    }


    //sync threads of the block so all populations are saved
    __syncthreads();

    //stream populations from other nodes towards the current node (pull)
    //define directions
    const unsigned short int tx = threadIdx.x;
    const unsigned short int ty = threadIdx.y;
    const unsigned short int tz = threadIdx.z;

    const unsigned short int bx = blockIdx.x;
    const unsigned short int by = blockIdx.y;
    const unsigned short int bz = blockIdx.z;

    // it is added the block size to get the populations from the other side, 
    //it will later be replaced with the populations from the interfarce
    const unsigned short int xp1 = (tx + 1 + BLOCK_NX)%BLOCK_NX;
    const unsigned short int xm1 = (tx - 1 + BLOCK_NX)%BLOCK_NX;

    const unsigned short int yp1 = (ty + 1 + BLOCK_NY)%BLOCK_NY;
    const unsigned short int ym1 = (ty - 1 + BLOCK_NY)%BLOCK_NY;

    const unsigned short int zp1 = (tz + 1 + BLOCK_NZ)%BLOCK_NZ;
    const unsigned short int zm1 = (tz - 1 + BLOCK_NZ)%BLOCK_NZ;

    //fStream[0 ] = stream_population[idxPopBlock(tx ,  ty,  tz,  1)]; // [idxPopBlock(tx ,  ty,  tz,  1)];
    fStream[1 ] = stream_population[idxPopBlock(xm1,  ty,  tz,  1)]; // [idxPopBlock(xp1,  ty,  tz,  1)];
    fStream[2 ] = stream_population[idxPopBlock(xp1,  ty,  tz,  2)]; // [idxPopBlock(xm1,  ty,  tz,  2)];
    fStream[3 ] = stream_population[idxPopBlock(tx,  ym1,  tz,  3)]; // [idxPopBlock(tx,  yp1,  tz,  3)];
    fStream[4 ] = stream_population[idxPopBlock(tx,  yp1,  tz,  4)]; // [idxPopBlock(tx,  ym1,  tz,  4)];
    fStream[5 ] = stream_population[idxPopBlock(tx,   ty, zm1,  5)]; // [idxPopBlock(tx,   ty, zp1,  5)];
    fStream[6 ] = stream_population[idxPopBlock(tx,   ty, zp1,  6)]; // [idxPopBlock(tx,   ty, zm1,  6)];
    fStream[7 ] = stream_population[idxPopBlock(xm1, ym1,  tz,  7)]; // [idxPopBlock(xp1, yp1,  tz,  7)];
    fStream[8 ] = stream_population[idxPopBlock(xp1, yp1,  tz,  8)]; // [idxPopBlock(xm1, ym1,  tz,  8)];
    fStream[9 ] = stream_population[idxPopBlock(xm1,  ty, zm1,  9)]; // [idxPopBlock(xp1,  ty, zp1,  9)];
    fStream[10] = stream_population[idxPopBlock(xp1,  ty, zp1, 10)]; // [idxPopBlock(xm1,  ty, zm1, 10)];
    fStream[11] = stream_population[idxPopBlock(tx,  ym1, zm1, 11)]; // [idxPopBlock(tx,  yp1, zp1, 11)];
    fStream[12] = stream_population[idxPopBlock(tx,  yp1, zp1, 12)]; // [idxPopBlock(tx,  ym1, zm1, 12)];
    fStream[13] = stream_population[idxPopBlock(xm1, yp1,  tz, 13)]; // [idxPopBlock(xp1, ym1,  tz, 13)];
    fStream[14] = stream_population[idxPopBlock(xp1, ym1,  tz, 14)]; // [idxPopBlock(xm1, yp1,  tz, 14)];
    fStream[15] = stream_population[idxPopBlock(xm1,  ty, zp1, 15)]; // [idxPopBlock(xp1,  ty, zm1, 15)];
    fStream[16] = stream_population[idxPopBlock(xp1,  ty, zm1, 16)]; // [idxPopBlock(xm1,  ty, zp1, 16)];
    fStream[17] = stream_population[idxPopBlock(tx,  ym1, zp1, 17)]; // [idxPopBlock(tx,  yp1, zm1, 17)];
    fStream[18] = stream_population[idxPopBlock(tx,  yp1, zm1, 18)]; // [idxPopBlock(tx,  ym1, zp1, 18)];
    #ifdef D3Q27    
    fStream[19] = stream_population[idxPopBlock(xm1, ym1, zm1, 19)]];; // [idxPopBlock(xp1, yp1, zp1, 19)]];
    fStream[20] = stream_population[idxPopBlock(xp1, yp1, zp1, 20)]];; // [idxPopBlock(xm1, ym1, zm1, 20)]];
    fStream[21] = stream_population[idxPopBlock(xm1, ym1, zp1, 21)]];; // [idxPopBlock(xp1, yp1, zm1, 21)]];
    fStream[22] = stream_population[idxPopBlock(xp1, yp1, zm1, 22)]];; // [idxPopBlock(xm1, ym1, zp1, 22)]];
    fStream[23] = stream_population[idxPopBlock(xm1, yp1, zm1, 23)]];; // [idxPopBlock(xp1, ym1, zp1, 23)]];
    fStream[24] = stream_population[idxPopBlock(xp1, ym1, zp1, 24)]];; // [idxPopBlock(xm1, yp1, zm1, 24)]];
    fStream[25] = stream_population[idxPopBlock(xp1, ym1, zm1, 25)]];; // [idxPopBlock(xm1, yp1, zp1, 25)]];
    fStream[26] = stream_population[idxPopBlock(xm1, yp1, zp1, 26)]];; // [idxPopBlock(xp1, ym1, zm1, 26)]];
    #endif

    // load populations from interface nodes

    
    if(tx == 0){ //check if is on west face of the block
        
        fStream[ 1] = pop.x[idxPopX(ty,tz,0,bx,by,bz)];
        fStream[ 7] = pop.x[idxPopX(ty,tz,1,bx,by,bz)];
        fStream[ 9] = pop.x[idxPopX(ty,tz,2,bx,by,bz)];
        fStream[13] = pop.x[idxPopX(ty,tz,3,bx,by,bz)];
        fStream[15] = pop.x[idxPopX(ty,tz,4,bx,by,bz)];
        
    }else if (tx == BLOCK_NX-1){ // check if is on east face

        fStream[ 2] = pop.x[idxPopX(ty,tz,5,bx,by,bz)];
        fStream[ 8] = pop.x[idxPopX(ty,tz,6,bx,by,bz)];
        fStream[10] = pop.x[idxPopX(ty,tz,7,bx,by,bz)];
        fStream[14] = pop.x[idxPopX(ty,tz,8,bx,by,bz)];
        fStream[16] = pop.x[idxPopX(ty,tz,9,bx,by,bz)];
    }
    if(ty == 0){ //check if is on south face of the block

        fStream[ 3] = pop.y[idxPopY(tx,tz,0,bx,by,bz)];
        fStream[ 7] = pop.y[idxPopY(tx,tz,1,bx,by,bz)];
        fStream[11] = pop.y[idxPopY(tx,tz,2,bx,by,bz)];
        fStream[14] = pop.y[idxPopY(tx,tz,3,bx,by,bz)];
        fStream[17] = pop.y[idxPopY(tx,tz,4,bx,by,bz)];

    }else if (ty == BLOCK_NY-1){ // check if is on north face
        
        fStream[ 4] = pop.y[idxPopX(tx,tz,5,bx,by,bz)];
        fStream[ 8] = pop.y[idxPopX(tx,tz,6,bx,by,bz)];
        fStream[12] = pop.y[idxPopX(tx,tz,7,bx,by,bz)];
        fStream[13] = pop.y[idxPopX(tx,tz,8,bx,by,bz)];
        fStream[18] = pop.y[idxPopX(tx,tz,9,bx,by,bz)];
        
    }
    if(tz == 0){ //check if is on back face of the block
    
        fStream[ 5] = pop.z[idxPopZ(tx,ty,0,bx,by,bz)];
        fStream[ 9] = pop.z[idxPopZ(tx,ty,1,bx,by,bz)];
        fStream[11] = pop.z[idxPopZ(tx,ty,2,bx,by,bz)];
        fStream[16] = pop.z[idxPopZ(tx,ty,3,bx,by,bz)];
        fStream[18] = pop.z[idxPopZ(tx,ty,4,bx,by,bz)];
    
    }else if (tz == BLOCK_NZ-1){ // check if is on front face
            
        fStream[ 6] = pop.z[idxPopZ(tx,ty,5,bx,by,bz)];
        fStream[10] = pop.z[idxPopZ(tx,ty,6,bx,by,bz)];
        fStream[12] = pop.z[idxPopZ(tx,ty,7,bx,by,bz)];
        fStream[15] = pop.z[idxPopZ(tx,ty,8,bx,by,bz)];
        fStream[17] = pop.z[idxPopZ(tx,ty,9,bx,by,bz)];
        
    }



    //compute new moments

    #ifdef D3Q19
    rhoVar = fStream[0] + fStream[1] + fStream[2] + fStream[3] + fStream[4] 
        + fStream[5] + fStream[6] + fStream[7] + fStream[8] + fStream[9] + fStream[10] 
        + fStream[11] + fStream[12] + fStream[13] + fStream[14] + fStream[15] + fStream[16] 
        + fStream[17] + fStream[18];
    dfloat invRho = 1/rhoVar;
    uxVar = ((fStream[1] + fStream[7] + fStream[9] + fStream[13] + fStream[15])
        - (fStream[2] + fStream[8] + fStream[10] + fStream[14] + fStream[16]) + 0.5*FX) * invRho;
    uyVar = ((fStream[3] + fStream[7] + fStream[11] + fStream[14] + fStream[17])
        - (fStream[4] + fStream[8] + fStream[12] + fStream[13] + fStream[18]) + 0.5*FY) * invRho;
    uzVar = ((fStream[5] + fStream[9] + fStream[11] + fStream[16] + fStream[18])
        - (fStream[6] + fStream[10] + fStream[12] + fStream[15] + fStream[17]) + 0.5*FZ) * invRho;
    #endif
    #ifdef D3Q27
    rhoVar = fStream[0] + fStream[1] + fStream[2] + fStream[3] + fStream[4] 
        + fStream[5] + fStream[6] + fStream[7] + fStream[8] + fStream[9] + fStream[10] 
        + fStream[11] + fStream[12] + fStream[13] + fStream[14] + fStream[15] + fStream[16] 
        + fStream[17] + fStream[18] + fStream[19] + fStream[20] + fStream[21] + fStream[22]
        + fStream[23] + fStream[24] + fStream[25] + fStream[26];
    const dfloat invRho = 1/rhoVar;
    uxVar = ((fStream[1] + fStream[7] + fStream[9] + fStream[13] + fStream[15]
        + fStream[19] + fStream[21] + fStream[23] + fStream[26]) 
        - (fStream[2] + fStream[8] + fStream[10] + fStream[14] + fStream[16] + fStream[20]
        + fStream[22] + fStream[24] + fStream[25]) + 0.5*fxVar) * invRho;
    uyVar = ((fStream[3] + fStream[7] + fStream[11] + fStream[14] + fStream[17]
        + fStream[19] + fStream[21] + fStream[24] + fStream[25])
        - (fStream[4] + fStream[8] + fStream[12] + fStream[13] + fStream[18] + fStream[20]
        + fStream[22] + fStream[23] + fStream[26]) + 0.5*fyVar) * invRho;
    uzVar = ((fStream[5] + fStream[9] + fStream[11] + fStream[16] + fStream[18]
        + fStream[19] + fStream[22] + fStream[23] + fStream[25])
        - (fStream[6] + fStream[10] + fStream[12] + fStream[15] + fStream[17] + fStream[20]
        + fStream[21] + fStream[24] + fStream[26]) + 0.5*fzVar) * invRho;
    #endif


    //Collide Moments
    //Equiblibrium momements
    dfloat mNodeEq[6];
    mNodeEq[0] = rhoVar * (uxVar * uxVar + cs2);
    mNodeEq[1] = rhoVar * (uxVar * uyVar);
    mNodeEq[2] = rhoVar * (uxVar * uzVar);
    mNodeEq[3] = rhoVar * (uyVar * uyVar + cs2);
    mNodeEq[4] = rhoVar * (uyVar * uzVar);
    mNodeEq[5] = rhoVar * (uzVar * uzVar + cs2);

    pxxVar = pxxVar - OMEGA*(pxxVar- mNodeEq[0]) + TT_OMEGA * (FX*uxVar + FX*uxVar);
    pxyVar = pxyVar - OMEGA*(pxyVar- mNodeEq[1]) + TT_OMEGA * (FX*uyVar + FY*uxVar);
    pxzVar = pxzVar - OMEGA*(pxzVar- mNodeEq[2]) + TT_OMEGA * (FX*uzVar + FZ*uxVar);
    pyyVar = pyyVar - OMEGA*(pyyVar- mNodeEq[3]) + TT_OMEGA * (FY*uyVar + FY*uyVar);
    pyzVar = pyzVar - OMEGA*(pyzVar- mNodeEq[4]) + TT_OMEGA * (FY*uzVar + FZ*uyVar);
    pzzVar = pzzVar - OMEGA*(pzzVar- mNodeEq[5]) + TT_OMEGA * (FZ*uzVar + FZ*uzVar);

    //compute new populations


    //compute macroscopics
    #ifdef D3Q19
    rhoVar = fStream[0] + fStream[1] + fStream[2] + fStream[3] + fStream[4] 
        + fStream[5] + fStream[6] + fStream[7] + fStream[8] + fStream[9] + fStream[10] 
        + fStream[11] + fStream[12] + fStream[13] + fStream[14] + fStream[15] + fStream[16] 
        + fStream[17] + fStream[18];
    invRho = 1/rhoVar;
    uxVar = ((fStream[1] + fStream[7] + fStream[9] + fStream[13] + fStream[15])
        - (fStream[2] + fStream[8] + fStream[10] + fStream[14] + fStream[16]) + 0.5*FX) * invRho;
    uyVar = ((fStream[3] + fStream[7] + fStream[11] + fStream[14] + fStream[17])
        - (fStream[4] + fStream[8] + fStream[12] + fStream[13] + fStream[18]) + 0.5*FY) * invRho;
    uzVar = ((fStream[5] + fStream[9] + fStream[11] + fStream[16] + fStream[18])
        - (fStream[6] + fStream[10] + fStream[12] + fStream[15] + fStream[17]) + 0.5*FZ) * invRho;
    #endif
    #ifdef D3Q27
    rhoVar = fStream[0] + fStream[1] + fStream[2] + fStream[3] + fStream[4] 
        + fStream[5] + fStream[6] + fStream[7] + fStream[8] + fStream[9] + fStream[10] 
        + fStream[11] + fStream[12] + fStream[13] + fStream[14] + fStream[15] + fStream[16] 
        + fStream[17] + fStream[18] + fStream[19] + fStream[20] + fStream[21] + fStream[22]
        + fStream[23] + fStream[24] + fStream[25] + fStream[26];
    const dfloat invRho = 1/rhoVar;
    uxVar = ((fStream[1] + fStream[7] + fStream[9] + fStream[13] + fStream[15]
        + fStream[19] + fStream[21] + fStream[23] + fStream[26]) 
        - (fStream[2] + fStream[8] + fStream[10] + fStream[14] + fStream[16] + fStream[20]
        + fStream[22] + fStream[24] + fStream[25]) + 0.5*fxVar) * invRho;
    uyVar = ((fStream[3] + fStream[7] + fStream[11] + fStream[14] + fStream[17]
        + fStream[19] + fStream[21] + fStream[24] + fStream[25])
        - (fStream[4] + fStream[8] + fStream[12] + fStream[13] + fStream[18] + fStream[20]
        + fStream[22] + fStream[23] + fStream[26]) + 0.5*fyVar) * invRho;
    uzVar = ((fStream[5] + fStream[9] + fStream[11] + fStream[16] + fStream[18]
        + fStream[19] + fStream[22] + fStream[23] + fStream[25])
        - (fStream[6] + fStream[10] + fStream[12] + fStream[15] + fStream[17] + fStream[20]
        + fStream[21] + fStream[24] + fStream[26]) + 0.5*fzVar) * invRho;
    #endif
    //write moments in global memory

    
    mom.rho[indexNodeLBM] = rhoVar;

    mom.ux[indexNodeLBM] =  uxVar;
    mom.uy[indexNodeLBM] =  uyVar;
    mom.uz[indexNodeLBM] =  uzVar;

    mom.pxx[indexNodeLBM] = pxxVar;
    mom.pxy[indexNodeLBM] = pxyVar;
    mom.pxz[indexNodeLBM] = pxzVar;
    mom.pyy[indexNodeLBM] = pyyVar;
    mom.pyz[indexNodeLBM] = pyzVar;
    mom.pzz[indexNodeLBM] = pzzVar;
    
    //write populations of the interface

if(tx == 0){ //check if is on west face of the block
        /*
        fPopWest[idxPopX(ty,tz,1,indexBlock)] = fStream[ 1];
        fPopWest[idxPopX(ty,tz,2,indexBlock)] = fStream[ 7];
        fPopWest[idxPopX(ty,tz,3,indexBlock)] = fStream[ 9];
        fPopWest[idxPopX(ty,tz,4,indexBlock)] = fStream[13];
        fPopWest[idxPopX(ty,tz,5,indexBlock)] = fStream[15];
        */
    }else if (tx == BLOCK_NX-1){ // check if is on east face
        /*
        fPopEast[idxPopX(ty,tz,1,indexBlock)] = fStream[ 2];
        fPopEast[idxPopX(ty,tz,2,indexBlock)] = fStream[ 8];
        fPopEast[idxPopX(ty,tz,3,indexBlock)] = fStream[10];
        fPopEast[idxPopX(ty,tz,4,indexBlock)] = fStream[14];
        fPopEast[idxPopX(ty,tz,5,indexBlock)] = fStream[16];
        */
    }
    if(ty == 0){ //check if is on south face of the block
        /*
        fPopSouth[idxPopY(tx,tz,1,indexBlock)] = fStream[ 3];
        fPopSouth[idxPopY(tx,tz,2,indexBlock)] = fStream[ 7];
        fPopSouth[idxPopY(tx,tz,3,indexBlock)] = fStream[11];
        fPopSouth[idxPopY(tx,tz,4,indexBlock)] = fStream[14];
        fPopSouth[idxPopY(tx,tz,5,indexBlock)] = fStream[17];
        */
    }else if (ty == BLOCK_NY-1){ // check if is on north face
        /*
        fPopNorth[idxPopX(tx,tz,1,indexBlock)] = fStream[ 4];
        fPopNorth[idxPopX(tx,tz,2,indexBlock)] = fStream[ 8];
        fPopNorth[idxPopX(tx,tz,3,indexBlock)] = fStream[12];
        fPopNorth[idxPopX(tx,tz,4,indexBlock)] = fStream[13];
        fPopNorth[idxPopX(tx,tz,5,indexBlock)] = fStream[18];
        */
    }
    if(tz == 0){ //check if is on back face of the block
    /*
        fPopBack[idxPopZ(tx,ty,1,indexBlock)] = fStream[ 5];
        fPopBack[idxPopZ(tx,ty,2,indexBlock)] = fStream[ 9];
        fPopBack[idxPopZ(tx,ty,3,indexBlock)] = fStream[11];
        fPopBack[idxPopZ(tx,ty,4,indexBlock)] = fStream[16];
        fPopBack[idxPopZ(tx,ty,5,indexBlock)] = fStream[18];
    */
    }else if (tz == BLOCK_NZ-1){ // check if is on front face
            /*
        fPopFront[idxPopZ(tx,ty,1,indexBlock)] = fStream[ 6];
        fPopFront[idxPopZ(tx,ty,2,indexBlock)] = fStream[10];
        fPopFront[idxPopZ(tx,ty,3,indexBlock)] = fStream[12];
        fPopFront[idxPopZ(tx,ty,4,indexBlock)] = fStream[15];
        fPopFront[idxPopZ(tx,ty,5,indexBlock)] = fStream[17];
        */
    }
    
    
}
