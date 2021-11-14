#include "lbmInitialization.cuh"


__global__
void gpuInitialization_mom(
    Moments mom)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);
    //printf("tx % d ty % d tz % d  bix %d biy %d biz %d --  x: %d y: %d z: %d idx %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, x, y, z, index);
    
    // zeroth moment
    mom.rho[index] = RHO_0;

    //first moments
    dfloat ux,uy,uz;

    ux = 0.0;
    uy = 0.0;
    uz = 0.0;

    mom.ux[index] = ux;
    mom.uy[index] = uy;
    mom.uz[index] = uz;

    //second moments
    //define equilibrium populations
    dfloat feq[Q];
    dfloat meq[3][3] = {0,0,0,0,0,0,0,0,0};
    char c1, c2;
    for (int i = 0; i < Q; i++) {
        feq[i] = gpu_f_eq(w[i] * RHO_0,
            3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
            1 - 1.5 * (ux * ux
                + uy * uy
                + uz * uz));
    }
    for (int i = 0; i < Q; i++) {
        for (int d1 = 0; d1 < 3; d1++) {
            if (d1 == 0) { //x
                c1 = cx[i];
            }
            if (d1 == 1) { //y
                c1 = cy[i];
            }
            if (d1 == 2) { //z
                c1 = cz[i];
            }
            if (c1 == 0) {
                continue;
            }
            for (int d2 = 0; d2 < 3; d2++) {
                if (d2 == 0) { //x
                    c2 = cx[i];
                }
                if (d2 == 1) { //y
                    c2 = cy[i];
                }
                if (d2 == 2) { //z
                    c2 = cz[i];
                }
                meq[d1][d2] = feq[i] * c1 * c2;
            }
        }
    }
    mom.pxx[index] = meq[1][1];
    mom.pxy[index] = meq[1][2];
    mom.pxz[index] = meq[1][3];
    mom.pyy[index] = meq[2][2];
    mom.pyz[index] = meq[2][3];
    mom.pzz[index] = meq[3][3];

}

__global__
void gpuInitialization_pop(
    Moments mom,
    Populations pop)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) // out of bounds
        return;

    // not near the interface
    if ( (threadIdx.x!=0 && threadIdx.x != (BLOCK_NX-1)) && (threadIdx.y!=0 && threadIdx.y != (BLOCK_NY-1)) && (threadIdx.z!=0 && threadIdx.y != (BLOCK_NZ-1)) )
        return;
    

    size_t index = idxScalarGlobal(x, y, z);

    dfloat rhoVar,uxVar,uyVar,uzVar;

    rhoVar = mom.rho[index];
    uxVar = mom.ux[index];
    uyVar = mom.uy[index];
    uzVar = mom.uz[index];

    dfloat fNode[Q];

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

    // Calculate equilibrium fNode
    fNode[0 ] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNode[1 ] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNode[2 ] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNode[3 ] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNode[4 ] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNode[5 ] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNode[6 ] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNode[7 ] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNode[8 ] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNode[9 ] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
    fNode[10] = gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15);
    fNode[11] = gpu_f_eq(rhoW2, uy3 + uz3, p1_muu15);
    fNode[12] = gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15);
    fNode[13] = gpu_f_eq(rhoW2, ux3 - uy3, p1_muu15);
    fNode[14] = gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15);
    fNode[15] = gpu_f_eq(rhoW2, ux3 - uz3, p1_muu15);
    fNode[16] = gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15);
    fNode[17] = gpu_f_eq(rhoW2, uy3 - uz3, p1_muu15);
    fNode[18] = gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15);
    #ifdef D3Q27
    fNode[19] = gpu_f_eq(rhoW3, ux3 + uy3 + uz3, p1_muu15);
    fNode[20] = gpu_f_eq(rhoW3, -ux3 - uy3 - uz3, p1_muu15);
    fNode[21] = gpu_f_eq(rhoW3, ux3 + uy3 - uz3, p1_muu15);
    fNode[22] = gpu_f_eq(rhoW3, -ux3 - uy3 + uz3, p1_muu15);
    fNode[23] = gpu_f_eq(rhoW3, ux3 - uy3 + uz3, p1_muu15);
    fNode[24] = gpu_f_eq(rhoW3, -ux3 + uy3 - uz3, p1_muu15);
    fNode[25] = gpu_f_eq(rhoW3, -ux3 + uy3 + uz3, p1_muu15);
    fNode[26] = gpu_f_eq(rhoW3, ux3 - uy3 - uz3, p1_muu15);
    #endif

    
    //0 dont need to
    //1




    if(threadIdx.x == 0){ //check if is on west face of the block
        pop.west[idxPopX(threadIdx.y,threadIdx.z,0,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 2];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,1,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 8];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,2,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[10];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,3,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[14];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,4,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[16];

    }else if (threadIdx.x == BLOCK_NX-1){ // check if is on east face
        pop.east[idxPopX(threadIdx.y,threadIdx.z,0,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 1];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,1,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 7];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,2,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 9];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,3,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[13];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,4,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[15];
    }


    if(threadIdx.y == 0){ //check if is on south face of the block
        pop.south[idxPopY(threadIdx.x,threadIdx.z,0,blockIdx.x,(blockIdx.y-1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[ 4];
        pop.south[idxPopY(threadIdx.x,threadIdx.z,1,blockIdx.x,(blockIdx.y-1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[ 8];
        pop.south[idxPopY(threadIdx.x,threadIdx.z,2,blockIdx.x,(blockIdx.y-1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[12];
        pop.south[idxPopY(threadIdx.x,threadIdx.z,3,blockIdx.x,(blockIdx.y-1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[13];
        pop.south[idxPopY(threadIdx.x,threadIdx.z,4,blockIdx.x,(blockIdx.y-1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[18];
    }else if (threadIdx.y == BLOCK_NY-1){ // check if is on north face
        pop.north[idxPopY(threadIdx.x,threadIdx.z,0,blockIdx.x,(blockIdx.y+1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[ 3];
        pop.north[idxPopY(threadIdx.x,threadIdx.z,1,blockIdx.x,(blockIdx.y+1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[ 7];
        pop.north[idxPopY(threadIdx.x,threadIdx.z,2,blockIdx.x,(blockIdx.y+1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[11];
        pop.north[idxPopY(threadIdx.x,threadIdx.z,3,blockIdx.x,(blockIdx.y+1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[14];
        pop.north[idxPopY(threadIdx.x,threadIdx.z,4,blockIdx.x,(blockIdx.y+1+NUM_BLOCK_Y)%NUM_BLOCK_Y,blockIdx.z)] = fNode[17];
    }


    if(threadIdx.z == 0){ //check if is on back face of the block
        //size_t aa = idxPopZ(threadIdx.x,threadIdx.y,0,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z);
        //printf("a index %d \n",aa); 
        pop.back[idxPopZ(threadIdx.x,threadIdx.y,0,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.back[idxPopZ(threadIdx.x,threadIdx.y,1,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.back[idxPopZ(threadIdx.x,threadIdx.y,2,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.back[idxPopZ(threadIdx.x,threadIdx.y,3,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.back[idxPopZ(threadIdx.x,threadIdx.y,4,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];

    }else if (threadIdx.z == BLOCK_NZ-1){ // check if is on front face
        //size_t bb = idxPopZ(threadIdx.x,threadIdx.y,0,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z);
        //printf("b index %d \n",bb);
        pop.front[idxPopZ(threadIdx.x,threadIdx.y,0,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.front[idxPopZ(threadIdx.x,threadIdx.y,1,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.front[idxPopZ(threadIdx.x,threadIdx.y,2,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.front[idxPopZ(threadIdx.x,threadIdx.y,3,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.front[idxPopZ(threadIdx.x,threadIdx.y,4,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }

    

}