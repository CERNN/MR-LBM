#include "lbmInitialization.cuh"

__global__ void gpuInitialization_mom(
    dfloat *fMom)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    const unsigned short int tx = threadIdx.x;
    const unsigned short int ty = threadIdx.y;
    const unsigned short int tz = threadIdx.z;

    const unsigned short int bx = blockIdx.x;
    const unsigned short int by = blockIdx.y;
    const unsigned short int bz = blockIdx.z;

    size_t index = idxScalarGlobal(x, y, z);
    //printf("tx % d ty % d tz % d  bix %d biy %d biz %d --  x: %d y: %d z: %d idx %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, x, y, z, index);

    //first moments
    dfloat ux, uy, uz;

    ux = 0.1;
    uy = 0.0;
    uz = 0.0;

    // zeroth moment
    fMom[idxMom(tx, ty, tz, 0, bx, by, bz)] = RHO_0;
    fMom[idxMom(tx, ty, tz, 1, bx, by, bz)] = ux;
    fMom[idxMom(tx, ty, tz, 2, bx, by, bz)] = uy;
    fMom[idxMom(tx, ty, tz, 3, bx, by, bz)] = uz;

    //second moments
    //define equilibrium populations
    //dfloat feq[Q];
    dfloat meq[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    /*char c1, c2;
    for (int i = 0; i < Q; i++)
    {
        feq[i] = gpu_f_eq(w[i] * RHO_0,
                          3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1 - 1.5 * (ux * ux + uy * uy + uz * uz));
    }
    for (int i = 0; i < Q; i++)
    {
        for (int d1 = 0; d1 < 3; d1++)
        {
            if (d1 == 0)
            { //x
                c1 = cx[i];
            }
            if (d1 == 1)
            { //y
                c1 = cy[i];
            }
            if (d1 == 2)
            { //z
                c1 = cz[i];
            }
            //if (c1 == 0)
            //{
            //    continue;
            //}
            for (int d2 = 0; d2 < 3; d2++)
            {
                if (d2 == 0)
                { //x
                    c2 = cx[i];
                }
                if (d2 == 1)
                { //y
                    c2 = cy[i];
                }
                if (d2 == 2)
                { //z
                    c2 = cz[i];
                }
                meq[d1][d2] += feq[i] * c1 * c2;
            }
        }
    }*/
    fMom[idxMom(tx, ty, tz, 4, bx, by, bz)] = RHO_0*ux*ux+RHO_0*cs2;
    fMom[idxMom(tx, ty, tz, 5, bx, by, bz)] = RHO_0*ux*uy;
    fMom[idxMom(tx, ty, tz, 6, bx, by, bz)] = RHO_0*ux*uz;
    fMom[idxMom(tx, ty, tz, 7, bx, by, bz)] = RHO_0*uy*uy+RHO_0*cs2;
    fMom[idxMom(tx, ty, tz, 8, bx, by, bz)] = RHO_0*uy*uz;
    fMom[idxMom(tx, ty, tz, 9, bx, by, bz)] = RHO_0*uz*uz+RHO_0*cs2;
}

__global__ void gpuInitialization_pop(
    dfloat *fMom,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1)
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

    dfloat rhoVar, uxVar, uyVar, uzVar;

    rhoVar = fMom[idxMom(tx, ty, tz, 0, bx, by, bz)];
    uxVar = fMom[idxMom(tx, ty, tz, 1, bx, by, bz)];
    uyVar = fMom[idxMom(tx, ty, tz, 2, bx, by, bz)];
    uzVar = fMom[idxMom(tx, ty, tz, 3, bx, by, bz)];

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
    fNode[ 0] = gpu_f_eq(rhoW0, 0, p1_muu15);
    fNode[ 1] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    fNode[ 2] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    fNode[ 3] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    fNode[ 4] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    fNode[ 5] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    fNode[ 6] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    fNode[ 7] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    fNode[ 8] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    fNode[ 9] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
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
    /*
    if(      ty == 0            && tx == 0              && tz == 0)             {//swb
        fGhostX_0[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 2];
        fGhostX_0[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        fGhostX_0[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        fGhostX_0[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        fGhostX_0[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        
        fGhostY_0[idxPopY(tx,tz,5,(bx+cx[ 4]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 4]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 4];
        fGhostY_0[idxPopY(tx,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        fGhostY_0[idxPopY(tx,tz,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        fGhostY_0[idxPopY(tx,tz,8,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        fGhostY_0[idxPopY(tx,tz,9,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
         
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 6]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(ty == 0            && tx == 0              && tz == (NUM_BLOCK_Z-1))  {//swf
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 4]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 5]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }else if(ty == 0            && tx == (NUM_BLOCK_X-1)   && tz == 0)             {//seb
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 1]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 1]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 4]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 6]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(ty == 0            && tx == (NUM_BLOCK_X-1)   && tz == (NUM_BLOCK_Z-1))  {//sef
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 1]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 1]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 4]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 5]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }else if(ty == (NUM_BLOCK_Y-1) && tx == 0              && tz == 0)             {//nwb
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 3]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 6]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(ty == (NUM_BLOCK_Y-1) && tx == 0              && tz == (NUM_BLOCK_Z-1))  {//nwf
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 3]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 5]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }else if(ty == (NUM_BLOCK_Y-1) && tx == (NUM_BLOCK_X-1)   && tz == 0)             {//neb
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 1]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 1]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 3]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 6]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(ty == (NUM_BLOCK_Y-1) && tx == (NUM_BLOCK_X-1)   && tz == (NUM_BLOCK_Z-1))  {//nef
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 1]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 1]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];

        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 3]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];

        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 5]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];



/* ------------------------------ EDGE ------------------------------ */
    /*

    }else if(ty == 0            && tx == 0)             {//sw
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[16];        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[18];
    }else if(ty == 0            && tx == (NUM_BLOCK_X-1))  {//se
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 1]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[15];        
        pop.y[idxPopY(tx,tz,5,(bx+cx[ 4]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[18];
    }else if(ty == (NUM_BLOCK_Y-1) && tx == 0)             {//nw
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 2]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[16];        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[17];
    }else if(ty == (NUM_BLOCK_Y-1) && tx == (NUM_BLOCK_X-1))  {//ne
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 1]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[15];        
        pop.y[idxPopY(tx,tz,0,(bx+cx[ 3]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz))] = fNode[17];
    }else if(ty == 0            && tz == 0)             {//sb
        pop.y[idxPopY(tx,tz,5,(bx),(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 4]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx),(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx),(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx),(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx),(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];        
        pop.z[idxPopZ(tx,ty,5,(bx),(by+cy[ 6]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx),(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx),(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx),(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx),(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];

    }else if(ty == 0            && tz == (NUM_BLOCK_Z-1))  {//sf
        
        pop.y[idxPopY(tx,tz,5,(bx),(by+cy[ 4]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 4]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 4];
        pop.y[idxPopY(tx,tz,6,(bx),(by+cy[ 8]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.y[idxPopY(tx,tz,7,(bx),(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.y[idxPopY(tx,tz,8,(bx),(by+cy[13]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.y[idxPopY(tx,tz,9,(bx),(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];        
        pop.z[idxPopZ(tx,ty,0,(bx),(by+cy[ 5]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx),(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx),(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx),(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx),(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }else if(ty == (NUM_BLOCK_Y-1) && tz == 0)             {//nb
        
        pop.y[idxPopY(tx,tz,0,(bx),(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 3]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx),(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx),(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx),(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx),(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];        
        pop.z[idxPopZ(tx,ty,5,(bx),(by+cy[ 6]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx),(by+cy[10]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx),(by+cy[12]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx),(by+cy[15]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx),(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(ty == (NUM_BLOCK_Y-1) && tz == (NUM_BLOCK_Z-1))  {//nf
        
        pop.y[idxPopY(tx,tz,0,(bx),(by+cy[ 3]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 3]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 3];
        pop.y[idxPopY(tx,tz,1,(bx),(by+cy[ 7]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.y[idxPopY(tx,tz,2,(bx),(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.y[idxPopY(tx,tz,3,(bx),(by+cy[14]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.y[idxPopY(tx,tz,4,(bx),(by+cy[17]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];        
        pop.z[idxPopZ(tx,ty,0,(bx),(by+cy[ 5]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx),(by+cy[ 9]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx),(by+cy[11]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx),(by+cy[16]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx),(by+cy[18]+NUM_BLOCK_Y)%NUM_BLOCK_Y,(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }else if(tx == 0            && tz == 0)             {//wb
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(tx == 0            && tz == (NUM_BLOCK_Z-1))  {//wf
        
        pop.x[idxPopX(ty,tz,5,(bx+cx[ 2]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 2]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 2];
        pop.x[idxPopX(ty,tz,6,(bx+cx[ 8]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 8]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 8];
        pop.x[idxPopX(ty,tz,7,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.x[idxPopX(ty,tz,8,(bx+cx[14]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[14]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[14];
        pop.x[idxPopX(ty,tz,9,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];
    }else if(tx == (NUM_BLOCK_X-1) && tz == 0)             {//eb
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 1]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];        
        pop.z[idxPopZ(tx,ty,5,(bx+cx[ 6]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 6]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 6];
        pop.z[idxPopZ(tx,ty,6,(bx+cx[10]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[10]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[10];
        pop.z[idxPopZ(tx,ty,7,(bx+cx[12]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[12]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[12];
        pop.z[idxPopZ(tx,ty,8,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];
        pop.z[idxPopZ(tx,ty,9,(bx+cx[17]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[17]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[17];
    }else if(tx == (NUM_BLOCK_X-1) && tz == (NUM_BLOCK_Z-1))  {//ef
        
        pop.x[idxPopX(ty,tz,0,(bx+cx[ 1]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 1]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 1];
        pop.x[idxPopX(ty,tz,1,(bx+cx[ 7]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 7]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 7];
        pop.x[idxPopX(ty,tz,2,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.x[idxPopX(ty,tz,3,(bx+cx[13]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[13]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[13];
        pop.x[idxPopX(ty,tz,4,(bx+cx[15]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[15]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[15];        
        pop.z[idxPopZ(tx,ty,0,(bx+cx[ 5]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 5]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 5];
        pop.z[idxPopZ(tx,ty,1,(bx+cx[ 9]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[ 9]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[ 9];
        pop.z[idxPopZ(tx,ty,2,(bx+cx[11]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[11]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[11];
        pop.z[idxPopZ(tx,ty,3,(bx+cx[16]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[16]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[16];
        pop.z[idxPopZ(tx,ty,4,(bx+cx[18]+NUM_BLOCK_X)%NUM_BLOCK_X,(by),(bz+cz[18]+NUM_BLOCK_Z)%NUM_BLOCK_Z)] = fNode[18];



/* ------------------------------ FACE ------------------------------ */
    if (tx == 0) { //w
        fGhostX_1[idxPopX(ty, tz, 0, (bx + cx[ 2] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[ 2];
        fGhostX_1[idxPopX(ty, tz, 1, (bx + cx[ 8] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[ 8];
        fGhostX_1[idxPopX(ty, tz, 2, (bx + cx[10] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[10];
        fGhostX_1[idxPopX(ty, tz, 3, (bx + cx[14] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[14];
        fGhostX_1[idxPopX(ty, tz, 4, (bx + cx[16] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[16];
    }else if (tx == (NUM_BLOCK_X - 1)){ //e                                                 -         
        fGhostX_0[idxPopX(ty, tz, 0, (bx + cx[ 1] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[ 1];
        fGhostX_0[idxPopX(ty, tz, 1, (bx + cx[ 7] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[ 7];
        fGhostX_0[idxPopX(ty, tz, 2, (bx + cx[ 9] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[ 9];
        fGhostX_0[idxPopX(ty, tz, 3, (bx + cx[13] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[13];
        fGhostX_0[idxPopX(ty, tz, 4, (bx + cx[15] + NUM_BLOCK_X) % NUM_BLOCK_X, (by), (bz))] = fNode[15];
    }if (ty == 0)  { //s                                                                                      
        fGhostY_1[idxPopY(tx, tz, 0, (bx), (by + cy[ 4] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[ 4];
        fGhostY_1[idxPopY(tx, tz, 1, (bx), (by + cy[ 8] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[ 8];
        fGhostY_1[idxPopY(tx, tz, 2, (bx), (by + cy[12] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[12];
        fGhostY_1[idxPopY(tx, tz, 3, (bx), (by + cy[13] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[13];
        fGhostY_1[idxPopY(tx, tz, 4, (bx), (by + cy[18] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[18];
    }else if (ty == (NUM_BLOCK_Y - 1)){ //n                                                       
        fGhostY_0[idxPopY(tx, tz, 0, (bx), (by + cy[ 3] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[ 3];
        fGhostY_0[idxPopY(tx, tz, 1, (bx), (by + cy[ 7] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[ 7];
        fGhostY_0[idxPopY(tx, tz, 2, (bx), (by + cy[11] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[11];
        fGhostY_0[idxPopY(tx, tz, 3, (bx), (by + cy[14] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[14];
        fGhostY_0[idxPopY(tx, tz, 4, (bx), (by + cy[17] + NUM_BLOCK_Y) % NUM_BLOCK_Y, (bz))] = fNode[17];
    }if (tz == 0){ //b                                                                                    
        fGhostZ_1[idxPopZ(tx, ty, 0, (bx), (by), (bz + cz[ 6] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[ 6];
        fGhostZ_1[idxPopZ(tx, ty, 1, (bx), (by), (bz + cz[10] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[10];
        fGhostZ_1[idxPopZ(tx, ty, 2, (bx), (by), (bz + cz[12] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[12];
        fGhostZ_1[idxPopZ(tx, ty, 3, (bx), (by), (bz + cz[15] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[15];
        fGhostZ_1[idxPopZ(tx, ty, 4, (bx), (by), (bz + cz[17] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[17];
    } else if (tz == (NUM_BLOCK_Z - 1)) { //f                                                             
        fGhostZ_0[idxPopZ(tx, ty, 0, (bx), (by), (bz + cz[ 5] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[ 5];
        fGhostZ_0[idxPopZ(tx, ty, 1, (bx), (by), (bz + cz[ 9] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[ 9];
        fGhostZ_0[idxPopZ(tx, ty, 2, (bx), (by), (bz + cz[11] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[11];
        fGhostZ_0[idxPopZ(tx, ty, 3, (bx), (by), (bz + cz[16] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[16];
        fGhostZ_0[idxPopZ(tx, ty, 4, (bx), (by), (bz + cz[18] + NUM_BLOCK_Z) % NUM_BLOCK_Z)] = fNode[18];
    }
}
