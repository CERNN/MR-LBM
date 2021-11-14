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
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);
    // zeroth moment

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bz = blockIdx.z;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z;

    unsigned int bxp1 = (blockIdx.x + 1) % NUM_BLOCK_X;//(bx+1>=NUM_BLOCK_X) ? 0 : bx+1;
    unsigned int bxm1 = (blockIdx.x - 1) % NUM_BLOCK_X;//(bx-1<0)? NUM_BLOCK_X-1 :bx-1;

    unsigned int byp1 = (blockIdx.y + 1) % NUM_BLOCK_Y;//(by+1>=NUM_BLOCK_Y) ? 0 : by+1;
    unsigned int bym1 = (blockIdx.y - 1) % NUM_BLOCK_Y;//(by-1<0)? NUM_BLOCK_Y-1 : by-1;

    unsigned int bzp1 = (blockIdx.z + 1) % NUM_BLOCK_Z;//(bz+1>=NUM_BLOCK_Z) ? 0 : bz+1;
    unsigned int bzm1 = (blockIdx.z - 1) % NUM_BLOCK_Z;//(bz-1<0)? NUM_BLOCK_Z-1 : bz-1;


    dfloat rhoVar, uxVar, uyVar, uzVar;

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
    fNode[0] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNode[1] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNode[2] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNode[3] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNode[4] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNode[5] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNode[6] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNode[7] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNode[8] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNode[9] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
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
    /* ------------------------------ CORNER ------------------------------ */
    if(      ty == 0            && tx == 0              && tz == 0)             {//swb
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by+cy[ 2]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 2]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+BLOCK_NX)%BLOCK_NX,(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 4]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
         
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+BLOCK_NX)%BLOCK_NX,(by+cy[ 6]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(ty == 0            && tx == 0              && tz == (BLOCK_NZ-1))  {//swf
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by+cy[ 2]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 2]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+BLOCK_NX)%BLOCK_NX,(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 4]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+BLOCK_NX)%BLOCK_NX,(by+cy[ 5]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }else if(ty == 0            && tx == (BLOCK_NX-1)   && tz == 0)             {//seb
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by+cy[ 1]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 1]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+BLOCK_NX)%BLOCK_NX,(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 4]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+BLOCK_NX)%BLOCK_NX,(by+cy[ 6]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(ty == 0            && tx == (BLOCK_NX-1)   && tz == (BLOCK_NZ-1))  {//sef
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by+cy[ 1]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 1]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+BLOCK_NX)%BLOCK_NX,(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 4]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+BLOCK_NX)%BLOCK_NX,(by+cy[ 5]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }else if(ty == (BLOCK_NY-1) && tx == 0              && tz == 0)             {//nwb
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by+cy[ 2]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 2]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+BLOCK_NX)%BLOCK_NX,(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 3]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+BLOCK_NX)%BLOCK_NX,(by+cy[ 6]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(ty == (BLOCK_NY-1) && tx == 0              && tz == (BLOCK_NZ-1))  {//nwf
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by+cy[ 2]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 2]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+BLOCK_NX)%BLOCK_NX,(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 3]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+BLOCK_NX)%BLOCK_NX,(by+cy[ 5]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }else if(ty == (BLOCK_NY-1) && tx == (BLOCK_NX-1)   && tz == 0)             {//neb
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by+cy[ 1]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 1]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+BLOCK_NX)%BLOCK_NX,(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 3]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+BLOCK_NX)%BLOCK_NX,(by+cy[ 6]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(ty == (BLOCK_NY-1) && tx == (BLOCK_NX-1)   && tz == (BLOCK_NZ-1))  {//nef
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by+cy[ 1]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 1]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];

        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+BLOCK_NX)%BLOCK_NX,(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 3]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];

        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+BLOCK_NX)%BLOCK_NX,(by+cy[ 5]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];



/* ------------------------------ EDGE ------------------------------ */


    }else if(ty == 0            && tx == 0)             {//sw
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by+cy[ 2]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[16];        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+BLOCK_NX)%BLOCK_NX,(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[18];
    }else if(ty == 0            && tx == (BLOCK_NX-1))  {//se
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by+cy[ 1]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[15];        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+BLOCK_NX)%BLOCK_NX,(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[18];
    }else if(ty == (BLOCK_NY-1) && tx == 0)             {//nw
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by+cy[ 2]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[16];        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+BLOCK_NX)%BLOCK_NX,(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[17];
    }else if(ty == (BLOCK_NY-1) && tx == (BLOCK_NX-1))  {//ne
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by+cy[ 1]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[15];        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+BLOCK_NX)%BLOCK_NX,(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[17];
    }else if(ty == 0            && tz == 0)             {//sb
        pop.y[idxPopY(tx,tz,5,(bx),(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 4]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx),(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx),(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx),(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx),(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];        
        pop.z[idxPopZ(tx,ty,5,(bx),(by+cy[ 6]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx),(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx),(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx),(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx),(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];

    }else if(ty == 0            && tz == (BLOCK_NZ-1))  {//sf
        
        pop.y[idxPopY(tx,tz,5,(bx),(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 4]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx),(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx),(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx),(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx),(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];        
        pop.z[idxPopZ(tx,ty,0,(bx),(by+cy[ 5]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx),(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx),(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx),(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx),(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }else if(ty == (BLOCK_NY-1) && tz == 0)             {//nb
        
        pop.y[idxPopY(tx,tz,0,(bx),(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 3]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx),(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx),(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx),(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx),(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];        
        pop.z[idxPopZ(tx,ty,5,(bx),(by+cy[ 6]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx),(by+cy[10]+BLOCK_NY)%BLOCK_NY,(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx),(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx),(by+cy[15]+BLOCK_NY)%BLOCK_NY,(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx),(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(ty == (BLOCK_NY-1) && tz == (BLOCK_NZ-1))  {//nf
        
        pop.y[idxPopY(tx,tz,0,(bx),(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 3]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx),(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx),(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx),(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx),(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];        
        pop.z[idxPopZ(tx,ty,0,(bx),(by+cy[ 5]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx),(by+cy[ 9]+BLOCK_NY)%BLOCK_NY,(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx),(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx),(by+cy[16]+BLOCK_NY)%BLOCK_NY,(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx),(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }else if(tx == 0            && tz == 0)             {//wb
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 2]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(tx == 0            && tz == (BLOCK_NZ-1))  {//wf
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 2]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 8]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[14]+BLOCK_NZ)%BLOCK_NZ)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }else if(tx == (BLOCK_NX-1) && tz == 0)             {//eb
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 1]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(tx == (BLOCK_NX-1) && tz == (BLOCK_NZ-1))  {//ef
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 1]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 7]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[13]+BLOCK_NZ)%BLOCK_NZ)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+BLOCK_NX)%BLOCK_NX,(by),(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];



/* ------------------------------ FACE ------------------------------ */



    }else if(ty == 0)             {//s
        pop.y[idxPopY(tx,tz,5,(bx),(by+cy[ 4]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx),(by+cy[ 8]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx),(by+cy[12]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx),(by+cy[13]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx),(by+cy[18]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[18];
    }else if(ty == (BLOCK_NY-1))  {//n
        pop.y[idxPopY(tx,tz,0,(bx),(by+cy[ 3]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx),(by+cy[ 7]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx),(by+cy[11]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx),(by+cy[14]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx),(by+cy[17]+BLOCK_NY)%BLOCK_NY,(bz))] = fNode[17];
    }else if(tx == 0)             {//w
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[16];
    }else if(tx == (BLOCK_NX-1))  {//e
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+BLOCK_NX)%BLOCK_NX,(by),(bz))] = fNode[15];
    }else if(tz == 0)             {//b
        pop.z[idxPopZ(tx,ty,5,(bx),(by),(bz+cz[ 6]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx),(by),(bz+cz[10]+BLOCK_NZ)%BLOCK_NZ)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx),(by),(bz+cz[12]+BLOCK_NZ)%BLOCK_NZ)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx),(by),(bz+cz[15]+BLOCK_NZ)%BLOCK_NZ)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx),(by),(bz+cz[17]+BLOCK_NZ)%BLOCK_NZ)] = fNode[17];
    }else if(tz == (BLOCK_NZ-1))  {//f
        pop.z[idxPopZ(tx,ty,0,(bx),(by),(bz+cz[ 5]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx),(by),(bz+cz[ 9]+BLOCK_NZ)%BLOCK_NZ)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx),(by),(bz+cz[11]+BLOCK_NZ)%BLOCK_NZ)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx),(by),(bz+cz[16]+BLOCK_NZ)%BLOCK_NZ)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx),(by),(bz+cz[18]+BLOCK_NZ)%BLOCK_NZ)] = fNode[18];
    }

        
        


}

/*
__global__
void gpuInitialization_pop(
    Moments mom,
    Populations pop)
{
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ) // out of bounds
        return;

    // not near the interface
    if ( (threadIdx.x!=0 && threadIdx.x != ((BLOCK_NX-1))) && (threadIdx.y!=0 && threadIdx.y != (BLOCK_NY-1)) && (threadIdx.z!=0 && threadIdx.y != ((BLOCK_NZ-1))) )
        return;
    

    unsigned int index = x + y * 128 + z * 128 * 128;

    unsigned int bx = (unsigned int)blockIdx.x;
    unsigned int by = (unsigned int)blockIdx.y;
    unsigned int bz = (unsigned int)blockIdx.z;

    unsigned int bxp1 = (bx + (size_t)1) % NUM_BLOCK_X;//(bx+1>=NUM_BLOCK_X) ? 0 : bx+1;
    unsigned int bxm1 = (bx - (size_t)1) % NUM_BLOCK_X;//(bx-1<0)? NUM_BLOCK_X-1 :bx-1;

    unsigned int byp1 = (by + (size_t)1) % NUM_BLOCK_Y;//(by+1>=NUM_BLOCK_Y) ? 0 : by+1;
    unsigned int bym1 = (by - (size_t)1) % NUM_BLOCK_Y;//(by-1<0)? NUM_BLOCK_Y-1 : by-1;

    unsigned int bzp1 = (bz + (size_t)1) % NUM_BLOCK_Z;//(bz+1>=NUM_BLOCK_Z) ? 0 : bz+1;
    unsigned int bzm1 = (bz - (size_t)1) % NUM_BLOCK_Z;//(bz-1<0)? NUM_BLOCK_Z-1 : bz-1;
    

    printf("x %d %d %d %d %d y: %d %d %d %d %d z: %d %d %d %d %d\n",bx,bxp1,bxp1-bx,bxm1,bx-bxm1,by,byp1,byp1-by,bym1,by-bym1,bz,bzp1,bzp1-bz,bzm1,bz-bzm1);
    /*
    

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

    
    if(threadIdx.x == (BLOCK_NX-1)){
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,0,bxp1,by,bz)]=fNode[ 1];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,1,bxp1,by,bz)]=fNode[ 7];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,2,bxp1,by,bz)]=fNode[ 9];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,3,bxp1,by,bz)]=fNode[13];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,4,bxp1,by,bz)]=fNode[15];

    }else if(threadIdx.x == 0){
        //printf("bi %d ty %d tz %d pop %d bx %d by %d bz %d \n",idxPopX(threadIdx.y,threadIdx.z,5,(blockIdx.x+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z),threadIdx.y,threadIdx.z,5,(blockIdx.x+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z);        
        //printf("bi %d ty %d tz %d pop %d bx %d by %d bz %d \n",idxPopX(threadIdx.y,threadIdx.z,6,(blockIdx.x+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z),threadIdx.y,threadIdx.z,6,(blockIdx.x+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z);        
        //printf("bi %d ty %d tz %d pop %d bx %d by %d bz %d \n",idxPopX(threadIdx.y,threadIdx.z,7,(blockIdx.x+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z),threadIdx.y,threadIdx.z,7,(blockIdx.x+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z);        
        //printf("bi %d ty %d tz %d pop %d bx %d by %d bz %d \n",idxPopX(threadIdx.y,threadIdx.z,8,(blockIdx.x+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z),threadIdx.y,threadIdx.z,8,(blockIdx.x+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z);        
        //printf("bi %d ty %d tz %d pop %d bx %d by %d bz %d \n",idxPopX(threadIdx.y,threadIdx.z,9,(blockIdx.x+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z),threadIdx.y,threadIdx.z,9,(blockIdx.x+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(blockIdx.y+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(blockIdx.z+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z);        
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,5,bxm1,by,bz)]=fNode[ 2];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,6,bxm1,by,bz)]=fNode[ 8];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,7,bxm1,by,bz)]=fNode[10];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,8,bxm1,by,bz)]=fNode[14];
        //pop.x[idxPopX(threadIdx.y,threadIdx.z,9,bxm1,by,bz)]=fNode[16];
    }



    /*
    if(threadIdx.x == 0){ //check if is on west face of the block
        pop.west[idxPopX(threadIdx.y,threadIdx.z,0,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 2];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,1,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 8];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,2,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[10];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,3,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[14];
        pop.west[idxPopX(threadIdx.y,threadIdx.z,4,(blockIdx.x-1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[16];

    }else if (threadIdx.x == (BLOCK_NX-1)){ // check if is on east face
        pop.east[idxPopX(threadIdx.y,threadIdx.z,0,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 1];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,1,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 7];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,2,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[ 9];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,3,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[13];
        pop.east[idxPopX(threadIdx.y,threadIdx.z,4,(blockIdx.x+1+NUM_BLOCK_X)%NUM_BLOCK_X,blockIdx.y,blockIdx.z)] = fNode[15];
    }


    if(threadIdx.y == 0){ //check if is on south face of the block

    }else if (threadIdx.y == BLOCK_NY-1){ // check if is on north face

    }


    if(threadIdx.z == 0){ //check if is on back face of the block
        //size_t aa = idxPopZ(threadIdx.x,threadIdx.y,0,blockIdx.x,blockIdx.y,(blockIdx.z-1+NUM_BLOCK_Z)%NUM_BLOCK_Z);
        //printf("a index %d \n",aa); 


    }else if (threadIdx.z == (BLOCK_NZ-1)){ // check if is on front face
        //size_t bb = idxPopZ(threadIdx.x,threadIdx.y,0,blockIdx.x,blockIdx.y,(blockIdx.z+1+NUM_BLOCK_Z)%NUM_BLOCK_Z);
        //printf("b index %d \n",bb);

    }

    

}*/