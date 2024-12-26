#include "mlbm_viscoelastic.cuh"

#ifdef A_XX_DIST
__global__ void gpuConformationXXCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat ANode[GQ];
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (GQ - 1)];
    #endif
    // Load moments from global memory
    //velocities
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    //
    dfloat AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat GxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat invAxx = 1/AxxVar;
    dfloat Axx_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Axx_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Axx_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
    dfloat Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
    dfloat Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);
    

    //if(x == 60 && y == 60 && z == 60)
    //    printf("step %d xx %f \n", step,GxxVar);


    #include COLREC_AXX_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    //overwrite values
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */
    ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

    /* load pop from global in cover nodes */

    #include "includeFiles/conformationTransport/popLoad_Axx.inc"


   if(nodeType != BULK){
        #include CASE_AXX_BC_DEF
    }else{
        AxxVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
        AxxVar = AxxVar + GxxVar;
        invAxx= 1.0/AxxVar;

        Axx_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
        Axx_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
        Axx_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
    }

    Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
    Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
    Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

    //calculate post collision populations
    #include COLREC_AXX_RECONSTRUCTION
    
    /* write to global mom */
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxxVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qz_t30;
    
    #include "includeFiles\conformationTransport\popSave_Axx.inc"

    /*
    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        if (INTERFACE_BC_WEST) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
        }if (INTERFACE_BC_EAST){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
        }if (INTERFACE_BC_SOUTH)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
        }if (INTERFACE_BC_NORTH){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
        }if (INTERFACE_BC_BACK){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
        }if (INTERFACE_BC_FRONT){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
        }
    #endif
    */
}
#endif

#ifdef A_XY_DIST
__global__ void gpuConformationXYCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat ANode[GQ];
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (GQ - 1)];
    #endif
    // Load moments from global memory
    //velocities
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    //
    dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat GxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat invAxy = 1/AxyVar;
    dfloat Axy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Axy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Axy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
    dfloat Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
    dfloat Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

        //if(x == 60 && y == 60 && z == 60)
        //printf("step %d xy %f \n", step,GxyVar);

    
    #include COLREC_AXY_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    //overwrite values
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */
    ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

    /* load pop from global in cover nodes */

    #include "includeFiles/conformationTransport/popLoad_Axy.inc"


   if(nodeType != BULK){
        #include CASE_AXY_BC_DEF
    }else{
        AxyVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
        AxyVar = AxyVar + GxyVar;
        invAxy= 1.0/AxyVar;

        Axy_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
        Axy_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
        Axy_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
    }

    Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
    Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
    Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

    //calculate post collision populations
    #include COLREC_AXY_RECONSTRUCTION
    
    /* write to global mom */
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxyVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qz_t30;
    
    #include "includeFiles\conformationTransport\popSave_Axy.inc"
    /*
    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        if (INTERFACE_BC_WEST) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
        }if (INTERFACE_BC_EAST){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
        }if (INTERFACE_BC_SOUTH)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
        }if (INTERFACE_BC_NORTH){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
        }if (INTERFACE_BC_BACK){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
        }if (INTERFACE_BC_FRONT){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
        }
    #endif
    */
}
#endif

#ifdef A_XZ_DIST
__global__ void gpuConformationXZCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat ANode[GQ];
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (GQ - 1)];
    #endif
    // Load moments from global memory
    //velocities
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    //
    dfloat AxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat GxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat invAxz = 1/AxzVar;
    dfloat Axz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Axz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Axz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
    dfloat Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
    dfloat Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

        //if(x == 60 && y == 60 && z == 60)
        //printf("step %d xz %f \n", step,GxzVar);

    
    #include COLREC_AXZ_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    //overwrite values
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */
    ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

    /* load pop from global in cover nodes */

    #include "includeFiles/conformationTransport/popLoad_Axz.inc"


   if(nodeType != BULK){
        #include CASE_AXZ_BC_DEF
    }else{
        AxzVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
        AxzVar = AxzVar + GxzVar;
        invAxz= 1.0/AxzVar;

        Axz_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
        Axz_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
        Axz_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
    }

    Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
    Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
    Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

    //calculate post collision populations
    #include COLREC_AXZ_RECONSTRUCTION
    
    /* write to global mom */
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxzVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qz_t30;
    
    #include "includeFiles\conformationTransport\popSave_Axz.inc"
    /*
    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        if (INTERFACE_BC_WEST) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
        }if (INTERFACE_BC_EAST){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
        }if (INTERFACE_BC_SOUTH)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
        }if (INTERFACE_BC_NORTH){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
        }if (INTERFACE_BC_BACK){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
        }if (INTERFACE_BC_FRONT){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
        }
    #endif
    */
}
#endif

#ifdef A_YY_DIST
__global__ void gpuConformationYYCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat ANode[GQ];
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (GQ - 1)];
    #endif
    // Load moments from global memory
    //velocities
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    //
    dfloat AyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat GyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat invAyy = 1/AyyVar;
    dfloat Ayy_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Ayy_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Ayy_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
    dfloat Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
    dfloat Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

        //if(x == 60 && y == 60 && z == 60)
        //printf("step %d yy %f \n", step,GyyVar);

    
    #include COLREC_AYY_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    //overwrite values
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */
    ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

    /* load pop from global in cover nodes */

    #include "includeFiles/conformationTransport/popLoad_Ayy.inc"


   if(nodeType != BULK){
        #include CASE_AYY_BC_DEF
    }else{
        AyyVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
        AyyVar = AyyVar + GyyVar;
        invAyy= 1.0/AyyVar;

        Ayy_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
        Ayy_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
        Ayy_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
    }

    Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
    Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
    Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

    //calculate post collision populations
    #include COLREC_AYY_RECONSTRUCTION
    
    /* write to global mom */
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AyyVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qz_t30;
    
    #include "includeFiles\conformationTransport\popSave_Ayy.inc"
    /*
    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        if (INTERFACE_BC_WEST) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
        }if (INTERFACE_BC_EAST){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
        }if (INTERFACE_BC_SOUTH)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
        }if (INTERFACE_BC_NORTH){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
        }if (INTERFACE_BC_BACK){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
        }if (INTERFACE_BC_FRONT){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
        }
    #endif
    */
}
#endif

#ifdef A_YZ_DIST
__global__ void gpuConformationYZCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat ANode[GQ];
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (GQ - 1)];
    #endif
    // Load moments from global memory
    //velocities
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    //
    dfloat AyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat GyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat invAyz = 1/AyzVar;
    dfloat Ayz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Ayz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Ayz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
    dfloat Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
    dfloat Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

        //if(x == 60 && y == 60 && z == 60)
        //printf("step %d yz %f \n", step,GyzVar);

    
    #include COLREC_AYZ_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    //overwrite values
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */
    ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

    /* load pop from global in cover nodes */

    #include "includeFiles/conformationTransport/popLoad_Ayz.inc"


   if(nodeType != BULK){
        #include CASE_AYZ_BC_DEF
    }else{
        AyzVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
        AyzVar = AyzVar + GyzVar;
        invAyz= 1.0/AyzVar;

        Ayz_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
        Ayz_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
        Ayz_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
    }

    Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
    Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
    Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

    //calculate post collision populations
    #include COLREC_AYZ_RECONSTRUCTION
    
    /* write to global mom */
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AyzVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qz_t30;
    
    #include "includeFiles\conformationTransport\popSave_Ayz.inc"
    /*
    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        if (INTERFACE_BC_WEST) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
        }if (INTERFACE_BC_EAST){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
        }if (INTERFACE_BC_SOUTH)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
        }if (INTERFACE_BC_NORTH){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
        }if (INTERFACE_BC_BACK){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
        }if (INTERFACE_BC_FRONT){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
        }
    #endif
    */
}
#endif


#ifdef A_ZZ_DIST
__global__ void gpuConformationZZCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat ANode[GQ];
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (GQ - 1)];
    #endif
    // Load moments from global memory
    //velocities
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
    dfloat ux_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    //
    dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat GzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat invAzz = 1/AzzVar;
    dfloat Azz_qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Azz_qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat Azz_qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
    dfloat Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
    dfloat Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

        //if(x == 60 && y == 60 && z == 60)
        //printf("step %d zz %f \n", step,GzzVar);

    
    #include COLREC_AZZ_RECONSTRUCTION

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int txm1 = (tx-1+BLOCK_NX)%BLOCK_NX;
    const int txp1 = (tx+1+BLOCK_NX)%BLOCK_NX;

    const int tym1 = (ty-1+BLOCK_NY)%BLOCK_NY;
    const int typ1 = (ty+1+BLOCK_NY)%BLOCK_NY;

    const int tzm1 = (tz-1+BLOCK_NZ)%BLOCK_NZ;
    const int tzp1 = (tz+1+BLOCK_NZ)%BLOCK_NZ;

    const int bxm1 = (bx-1+NUM_BLOCK_X)%NUM_BLOCK_X;
    const int bxp1 = (bx+1+NUM_BLOCK_X)%NUM_BLOCK_X;

    const int bym1 = (by-1+NUM_BLOCK_Y)%NUM_BLOCK_Y;
    const int byp1 = (by+1+NUM_BLOCK_Y)%NUM_BLOCK_Y;

    const int bzm1 = (bz-1+NUM_BLOCK_Z)%NUM_BLOCK_Z;
    const int bzp1 = (bz+1+NUM_BLOCK_Z)%NUM_BLOCK_Z;

    //overwrite values
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = ANode[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = ANode[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = ANode[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = ANode[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = ANode[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = ANode[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = ANode[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = ANode[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = ANode[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = ANode[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = ANode[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = ANode[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = ANode[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = ANode[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = ANode[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = ANode[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = ANode[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = ANode[18];

    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */
    ANode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    ANode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    ANode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    ANode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    ANode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    ANode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    ANode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    ANode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    ANode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    ANode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    ANode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    ANode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    ANode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    ANode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    ANode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    ANode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    ANode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    ANode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];

    /* load pop from global in cover nodes */

    #include "includeFiles/conformationTransport/popLoad_Azz.inc"


   if(nodeType != BULK){
        #include CASE_AZZ_BC_DEF
    }else{
        AzzVar = ANode[0] + ANode[1] + ANode[2] + ANode[3] + ANode[4] + ANode[5] + ANode[6] + ANode[7] + ANode[8] + ANode[9] + ANode[10] + ANode[11] + ANode[12] + ANode[13] + ANode[14] + ANode[15] + ANode[16] + ANode[17] + ANode[18];
        AzzVar = AzzVar + GzzVar;
        invAzz= 1.0/AzzVar;

        Azz_qx_t30 = F_M_I_SCALE*((ANode[1] - ANode[2] + ANode[7] - ANode[ 8] + ANode[ 9] - ANode[10] + ANode[13] - ANode[14] + ANode[15] - ANode[16]));
        Azz_qy_t30 = F_M_I_SCALE*((ANode[3] - ANode[4] + ANode[7] - ANode[ 8] + ANode[11] - ANode[12] + ANode[14] - ANode[13] + ANode[17] - ANode[18]));
        Azz_qz_t30 = F_M_I_SCALE*((ANode[5] - ANode[6] + ANode[9] - ANode[10] + ANode[11] - ANode[12] + ANode[16] - ANode[15] + ANode[18] - ANode[17]));
    }

    Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
    Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
    Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

    //calculate post collision populations
    #include COLREC_AZZ_RECONSTRUCTION
    
    /* write to global mom */
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AzzVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qz_t30;
    
    #include "includeFiles\conformationTransport\popSave_Azz.inc"
    /*
    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        if (INTERFACE_BC_WEST) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
        }if (INTERFACE_BC_EAST){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
        }if (INTERFACE_BC_SOUTH)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
        }if (INTERFACE_BC_NORTH){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
        }if (INTERFACE_BC_BACK){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
        }if (INTERFACE_BC_FRONT){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
        }
    #endif
    */
}
#endif
