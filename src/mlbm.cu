#include "mlbm.cuh"

__global__ void gpuMomCollisionStream(
    dfloat *fMom, unsigned int *dNodeType,ghostInterfaceData ghostInterface,
    DENSITY_CORRECTION_PARAMS_DECLARATION(d_)
    BC_FORCES_PARAMS_DECLARATION(d_)
    unsigned int step,
    bool save)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    dfloat pop[Q];
    #ifdef CONVECTION_DIFFUSION_TRANSPORT
    dfloat gNode[GQ];
    #endif
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[MAX_SHARED_MEMORY_SIZE];
    #endif
    
    const int baseIdx = idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z);
    const int baseIdxPop = idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0);

    // Load moments from global memory

    //rho'
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;

    dfloat rhoVar = RHO_0 + fMom[baseIdx + M_RHO_INDEX];
    dfloat ux_t30     = fMom[baseIdx + BLOCK_LBM_SIZE * M_UX_INDEX];
    dfloat uy_t30     = fMom[baseIdx + BLOCK_LBM_SIZE * M_UY_INDEX];
    dfloat uz_t30     = fMom[baseIdx + BLOCK_LBM_SIZE * M_UZ_INDEX];
    dfloat m_xx_t45   = fMom[baseIdx + BLOCK_LBM_SIZE * M_MXX_INDEX];
    dfloat m_xy_t90   = fMom[baseIdx + BLOCK_LBM_SIZE * M_MXY_INDEX];
    dfloat m_xz_t90   = fMom[baseIdx + BLOCK_LBM_SIZE * M_MXZ_INDEX];
    dfloat m_yy_t45   = fMom[baseIdx + BLOCK_LBM_SIZE * M_MYY_INDEX];
    dfloat m_yz_t90   = fMom[baseIdx + BLOCK_LBM_SIZE * M_MYZ_INDEX];
    dfloat m_zz_t45   = fMom[baseIdx + BLOCK_LBM_SIZE * M_MZZ_INDEX];

    #ifdef OMEGA_FIELD
        //dfloat omegaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat omegaVar = fMom[baseIdx + BLOCK_LBM_SIZE * M_OMEGA_INDEX];
        dfloat t_omegaVar = 1 - omegaVar;
        dfloat tt_omegaVar = 1 - omegaVar/2;
        dfloat omegaVar_d2 = omegaVar / 2;
        dfloat tt_omega_t3 = tt_omegaVar * 3;
    #else
        const dfloat omegaVar = OMEGA;
        const dfloat t_omegaVar = 1 - omegaVar;
        const dfloat tt_omegaVar = 1 - omegaVar/2;
        const dfloat omegaVar_d2 = omegaVar / 2;
        const dfloat tt_omega_t3 = tt_omegaVar * 3;
    #endif
    
    /*
    if(z > (NZ_TOTAL-50)){
        dfloat dist = (z - (NZ_TOTAL-50))/((NZ_TOTAL)- (NZ_TOTAL-50));
        dfloat ttau = 0.5+ 3*VISC*(1000.0*dist*dist*dist+1.0);
        omegaVar = 1/ttau;
    }*/

    //Local forces
    //dfloat K_const = 2.0*M_PI/(dfloat)N;
   // dfloat xx = 2.0 * M_PI * x / L;
   // dfloat yy = 2.0 * M_PI * y / L;
   // dfloat zz = 2.0 * M_PI * z / L;

   dfloat L_Fx = 0.00; // F_0 * sin(K_const*x) * cos(K_const*y) ;
   dfloat L_Fy = 0.00; //-F_0 * sin(K_const*y) * cos(K_const*x) ;
   dfloat L_Fz = 0.00;

    #ifdef BC_FORCES
    dfloat L_BC_Fx = 0.0;
    dfloat L_BC_Fy = 0.0;
    dfloat L_BC_Fz = 0.0;
    #endif


    #include COLREC_RECONSTRUCTION

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

    //need to compute the gradient before the moments are recalculated
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
        #include "includeFiles/velocity_gradient.inc"
    #endif //COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

    
    #ifdef CONFORMATION_TENSOR
        #ifdef A_XX_DIST
            dfloat AxxVar = fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_C_INDEX];
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            dfloat AxyVar = fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_C_INDEX];
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            dfloat AxzVar = fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_C_INDEX];
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            dfloat AyyVar = fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_C_INDEX];
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            dfloat AyzVar = fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_C_INDEX];
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            dfloat AzzVar = fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_C_INDEX];
        #endif //A_ZZ_DIST

        #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
            #include "includeFiles/conformationTransport/conformation_gradient.inc"   
        #endif

        #include "includeFiles/conformationTransport/conformation_evolution.inc"
    #endif

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        #ifdef SECOND_DIST 
            dfloat cVar = fMom[baseIdx + BLOCK_LBM_SIZE * M2_C_INDEX];
            dfloat invC = 1/cVar;
            dfloat qx_t30   = fMom[baseIdx + BLOCK_LBM_SIZE * M2_CX_INDEX];
            dfloat qy_t30   = fMom[baseIdx + BLOCK_LBM_SIZE * M2_CY_INDEX];
            dfloat qz_t30   = fMom[baseIdx + BLOCK_LBM_SIZE * M2_CZ_INDEX];

            dfloat udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - ux_t30);
            dfloat udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uy_t30);
            dfloat udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uz_t30);

            #include  COLREC_G_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */        
            #include "includeFiles/g_popLoad.inc"


            if(nodeType != BULK){
                #include CASE_G_BC_DEF
            }else{
                cVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                cVar = cVar + T_Q_INTERNAL_D_Cp;
                invC= 1.0/cVar;

                qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif
        #ifdef A_XX_DIST
            dfloat GxxVar = fMom[baseIdx + BLOCK_LBM_SIZE * G_XX_C_INDEX];
            dfloat invAxx = 1/AxxVar;
            dfloat Axx_qx_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_CX_INDEX];
            dfloat Axx_qy_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_CY_INDEX];
            dfloat Axx_qz_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_CZ_INDEX];


            dfloat Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
            dfloat Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
            dfloat Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

            #include COLREC_AXX_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            #include "includeFiles/conformationTransport/popLoad_Axx.inc"

            if(nodeType != BULK){
                 #include CASE_AXX_BC_DEF
            }else{
                AxxVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AxxVar = AxxVar + GxxVar;
                invAxx= 1.0/AxxVar;

                Axx_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Axx_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Axx_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            dfloat GxyVar = fMom[baseIdx + BLOCK_LBM_SIZE * G_XY_C_INDEX];
            dfloat invAxy = 1/AxyVar;
            dfloat Axy_qx_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_CX_INDEX];
            dfloat Axy_qy_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_CY_INDEX];
            dfloat Axy_qz_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_CZ_INDEX];


            dfloat Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
            dfloat Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
            dfloat Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

            #include COLREC_AXY_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            #include "includeFiles/conformationTransport/popLoad_Axy.inc"

            if(nodeType != BULK){
                    #include CASE_AXY_BC_DEF
            }else{
                AxyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AxyVar = AxyVar + GxyVar;
                invAxy= 1.0/AxyVar;

                Axy_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Axy_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Axy_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            dfloat GxzVar = fMom[baseIdx + BLOCK_LBM_SIZE * G_XZ_C_INDEX];
            dfloat invAxz = 1/AxzVar;
            dfloat Axz_qx_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_CX_INDEX];
            dfloat Axz_qy_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_CY_INDEX];
            dfloat Axz_qz_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_CZ_INDEX];


            dfloat Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
            dfloat Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
            dfloat Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

            #include COLREC_AXZ_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            #include "includeFiles/conformationTransport/popLoad_Axz.inc"

            if(nodeType != BULK){
                    #include CASE_AXZ_BC_DEF
            }else{
                AxzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AxzVar = AxzVar + GxzVar;
                invAxz= 1.0/AxzVar;

                Axz_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Axz_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Axz_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            dfloat GyyVar = fMom[baseIdx + BLOCK_LBM_SIZE * G_YY_C_INDEX];
            dfloat invAyy = 1/AyyVar;
            dfloat Ayy_qx_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_CX_INDEX];
            dfloat Ayy_qy_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_CY_INDEX];
            dfloat Ayy_qz_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_CZ_INDEX];


            dfloat Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
            dfloat Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
            dfloat Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

            #include COLREC_AYY_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            #include "includeFiles/conformationTransport/popLoad_Ayy.inc"

            if(nodeType != BULK){
                    #include CASE_AYY_BC_DEF
            }else{
                AyyVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AyyVar = AyyVar + GyyVar;
                invAyy= 1.0/AyyVar;

                Ayy_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Ayy_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Ayy_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            dfloat GyzVar = fMom[baseIdx + BLOCK_LBM_SIZE * G_YZ_C_INDEX];
            dfloat invAyz = 1/AyzVar;
            dfloat Ayz_qx_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_CX_INDEX];
            dfloat Ayz_qy_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_CY_INDEX];
            dfloat Ayz_qz_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_CZ_INDEX];


            dfloat Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
            dfloat Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
            dfloat Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

            #include COLREC_AYZ_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            #include "includeFiles/conformationTransport/popLoad_Ayz.inc"

            if(nodeType != BULK){
                    #include CASE_AYZ_BC_DEF
            }else{
                AyzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AyzVar = AyzVar + GyzVar;
                invAyz= 1.0/AyzVar;

                Ayz_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Ayz_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Ayz_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            dfloat GzzVar = fMom[baseIdx + BLOCK_LBM_SIZE * G_ZZ_C_INDEX];
            dfloat invAzz = 1/AzzVar;
            dfloat Azz_qx_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_CX_INDEX];
            dfloat Azz_qy_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_CY_INDEX];
            dfloat Azz_qz_t30 = fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_CZ_INDEX];


            dfloat Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
            dfloat Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
            dfloat Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

            #include COLREC_AZZ_RECONSTRUCTION

            __syncthreads();

            #include "includeFiles/convection_diffusion_streaming.inc"
            /* load pop from global in cover nodes */
            #include "includeFiles/conformationTransport/popLoad_Azz.inc"

            if(nodeType != BULK){
                    #include CASE_AZZ_BC_DEF
            }else{
                AzzVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
                AzzVar = AzzVar + GzzVar;
                invAzz= 1.0/AzzVar;

                Azz_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                Azz_qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                Azz_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
            }
        #endif //A_ZZ_DIST
        

    #endif //CONVECTION_DIFFUSION_TRANSPORT

    //save populations in shared memory

    s_pop[baseIdxPop +  0*BLOCK_LBM_SIZE] = pop[ 1];
    s_pop[baseIdxPop +  1*BLOCK_LBM_SIZE] = pop[ 2];
    s_pop[baseIdxPop +  2*BLOCK_LBM_SIZE] = pop[ 3];
    s_pop[baseIdxPop +  3*BLOCK_LBM_SIZE] = pop[ 4];
    s_pop[baseIdxPop +  4*BLOCK_LBM_SIZE] = pop[ 5];
    s_pop[baseIdxPop +  5*BLOCK_LBM_SIZE] = pop[ 6];
    s_pop[baseIdxPop +  6*BLOCK_LBM_SIZE] = pop[ 7];
    s_pop[baseIdxPop +  7*BLOCK_LBM_SIZE] = pop[ 8];
    s_pop[baseIdxPop +  8*BLOCK_LBM_SIZE] = pop[ 9];
    s_pop[baseIdxPop +  9*BLOCK_LBM_SIZE] = pop[10];
    s_pop[baseIdxPop + 10*BLOCK_LBM_SIZE] = pop[11];
    s_pop[baseIdxPop + 11*BLOCK_LBM_SIZE] = pop[12];
    s_pop[baseIdxPop + 12*BLOCK_LBM_SIZE] = pop[13];
    s_pop[baseIdxPop + 13*BLOCK_LBM_SIZE] = pop[14];
    s_pop[baseIdxPop + 14*BLOCK_LBM_SIZE] = pop[15];
    s_pop[baseIdxPop + 15*BLOCK_LBM_SIZE] = pop[16];
    s_pop[baseIdxPop + 16*BLOCK_LBM_SIZE] = pop[17];
    s_pop[baseIdxPop + 17*BLOCK_LBM_SIZE] = pop[18];
    #ifdef D3Q27
    s_pop[baseIdxPop + 18*BLOCK_LBM_SIZE] = pop[19];
    s_pop[baseIdxPop + 19*BLOCK_LBM_SIZE] = pop[20];
    s_pop[baseIdxPop + 20*BLOCK_LBM_SIZE] = pop[21];
    s_pop[baseIdxPop + 21*BLOCK_LBM_SIZE] = pop[22];
    s_pop[baseIdxPop + 22*BLOCK_LBM_SIZE] = pop[23];
    s_pop[baseIdxPop + 23*BLOCK_LBM_SIZE] = pop[24];
    s_pop[baseIdxPop + 24*BLOCK_LBM_SIZE] = pop[25];
    s_pop[baseIdxPop + 25*BLOCK_LBM_SIZE] = pop[26];
    #endif //D3Q27


    //sync threads of the block so all populations are saved
    __syncthreads();

    /* pull */

    pop[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
    pop[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
    pop[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
    pop[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
    pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
    pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
    pop[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
    pop[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
    pop[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
    pop[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
    pop[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
    pop[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
    pop[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
    pop[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
    pop[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
    pop[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
    pop[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
    pop[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];
    #ifdef D3Q27
    pop[19] = s_pop[idxPopBlock(xm1, ym1, zm1, 18)];
    pop[20] = s_pop[idxPopBlock(xp1, yp1, zp1, 19)];
    pop[21] = s_pop[idxPopBlock(xm1, ym1, zp1, 20)];
    pop[22] = s_pop[idxPopBlock(xp1, yp1, zm1, 21)];
    pop[23] = s_pop[idxPopBlock(xm1, yp1, zm1, 22)];
    pop[24] = s_pop[idxPopBlock(xp1, ym1, zp1, 23)];
    pop[25] = s_pop[idxPopBlock(xp1, ym1, zm1, 24)];
    pop[26] = s_pop[idxPopBlock(xm1, yp1, zp1, 25)];
    #endif

    /* load pop from global in cover nodes */

    #include "includeFiles/popLoad.inc"

    dfloat invRho;
    if(nodeType != BULK){
        #include CASE_BC_DEF

        invRho = 1.0 / rhoVar;               
    }else{

        //calculate streaming moments
        #ifdef D3Q19
            //equation3
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
            invRho = 1 / rhoVar;
            //equation4 + force correction
            ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[ 8] + pop[ 9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16])) * invRho;
            uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[ 8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18])) * invRho;
            uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17])) * invRho;

            //equation5
            m_xx_t45 = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16])* invRho - cs2;
            m_xy_t90 = (pop[7] - pop[13] + pop[8] - pop[14])* invRho;
            m_xz_t90 = (pop[9] - pop[15] + pop[10] - pop[16])* invRho;
            m_yy_t45 = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18])* invRho - cs2;
            m_yz_t90 = (pop[11] - pop[17] + pop[12] - pop[18])* invRho;
            m_zz_t45 = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18])* invRho - cs2;


        #endif
        #ifdef D3Q27
            rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
            invRho = 1 / rhoVar;
            ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25])) * invRho;
            uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26])) * invRho;
            uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26])) * invRho;

            m_xx_t45 = ( (pop[ 1] + pop[ 2] + pop[ 7] + pop[ 8] + pop[ 9] + pop[10]  +  pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
            m_xy_t90 = (((pop[ 7] + pop[ 8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])) )* invRho;
            m_xz_t90 = (((pop[ 9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])) )* invRho;
            m_yy_t45 = ( (pop[ 3] + pop[ 4] + pop[ 7] + pop[ 8] + pop[11] + pop[12]  +  pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
            m_yz_t90 = (((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])))* invRho;
            m_zz_t45 = ( (pop[ 5] + pop[ 6] + pop[ 9] + pop[10] + pop[11] + pop[12]  +  pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]))* invRho - cs2;
        #endif
    }

    // multiply moments by as2 -- as4*0.5 -- as4 - add correction to m_alpha_beta
    ux_t30 = F_M_I_SCALE * ux_t30;
    uy_t30 = F_M_I_SCALE * uy_t30;
    uz_t30 = F_M_I_SCALE * uz_t30;

    m_xx_t45 = F_M_II_SCALE * (m_xx_t45);
    m_xy_t90 = F_M_IJ_SCALE * (m_xy_t90);
    m_xz_t90 = F_M_IJ_SCALE * (m_xz_t90);
    m_yy_t45 = F_M_II_SCALE * (m_yy_t45);
    m_yz_t90 = F_M_IJ_SCALE * (m_yz_t90);
    m_zz_t45 = F_M_II_SCALE * (m_zz_t45);


    #ifdef DENSITY_CORRECTION
        //printf("%f ",d_mean_rho[0]-1.0) ;
        rhoVar -= (d_mean_rho[0]) ;
        invRho = 1/rhoVar;
    #endif // DENSITY_CORRECTION
    #ifdef THERMAL_MODEL //Boussinesq Approximation
        if(nodeType == BULK && T_BOUYANCY){
                L_Fx += gravity_vector[0] * T_gravity_t_beta * RHO_0*((cVar-T_REFERENCE));
                L_Fy += gravity_vector[1] * T_gravity_t_beta * RHO_0*((cVar-T_REFERENCE));
                L_Fz += gravity_vector[2] * T_gravity_t_beta * RHO_0*((cVar-T_REFERENCE));
        }
            
    #endif
    
    #ifdef COMPUTE_SHEAR
            //TODO change to fix perfomance
        const dfloat S_XX = rhoVar * (m_xx_t45/F_M_II_SCALE - ux_t30*ux_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_YY = rhoVar * (m_yy_t45/F_M_II_SCALE - uy_t30*uy_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_ZZ = rhoVar * (m_zz_t45/F_M_II_SCALE - uz_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_XY = rhoVar * (m_xy_t90/F_M_IJ_SCALE - ux_t30*uy_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_XZ = rhoVar * (m_xz_t90/F_M_IJ_SCALE - ux_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));
        const dfloat S_YZ = rhoVar * (m_yz_t90/F_M_IJ_SCALE - uy_t30*uz_t30/(F_M_I_SCALE*F_M_I_SCALE));

        const dfloat uFxxd2 = ux_t30*L_Fx/F_M_I_SCALE; // d2 = uFxx Divided by two
        const dfloat uFyyd2 = uy_t30*L_Fy/F_M_I_SCALE;
        const dfloat uFzzd2 = uz_t30*L_Fz/F_M_I_SCALE;
        const dfloat uFxyd2 = (ux_t30*L_Fy + uy_t30*L_Fx) / (2.0*F_M_I_SCALE);
        const dfloat uFxzd2 = (ux_t30*L_Fz + uz_t30*L_Fx) / (2.0*F_M_I_SCALE);
        const dfloat uFyzd2 = (uy_t30*L_Fz + uz_t30*L_Fy) / (2.0*F_M_I_SCALE);

        const dfloat auxStressMag = sqrt(0.5 * (
            (S_XX + uFxxd2) * (S_XX + uFxxd2) +(S_YY + uFyyd2) * (S_YY + uFyyd2) + (S_ZZ + uFzzd2) * (S_ZZ + uFzzd2) +
            2 * ((S_XY + uFxyd2) * (S_XY + uFxyd2) + (S_XZ + uFxzd2) * (S_XZ + uFxzd2) + (S_YZ + uFyzd2) * (S_YZ + uFyzd2))));

    #endif
    // MOMENTS DETERMINED, COMPUTE OMEGA IF NON-NEWTONIAN FLUID
    #if defined(OMEGA_FIELD)
            #ifdef NON_NEWTONIAN_FLUID 
                omegaVar = calcOmega_nnf(omegaVar, auxStressMag,step);
            #endif//  OMEGA_FIELD

            #ifdef LES_MODEL
                dfloat tau_t = 0.5*sqrt(TAU*TAU+Implicit_const*auxStressMag)-0.5*TAU;
                dfloat visc_turb_var = tau_t/3.0;

                omegaVar = 1.0/(TAU + tau_t);
            #endif

            //Compute new auxiliary variables
            t_omegaVar = 1 - omegaVar;
            tt_omegaVar = 1 - 0.5*omegaVar;
            omegaVar_d2 = omegaVar / 2.0;
            tt_omega_t3 = tt_omegaVar * 3.0;
    #endif 
    
        // COLLIDE
    #include COLREC_COLLISION
    

    //calculate post collision populations
    #include COLREC_RECONSTRUCTION
    
    
    /* write to global mom */

    fMom[baseIdx + M_RHO_INDEX] = rhoVar - RHO_0;

    fMom[baseIdx + BLOCK_LBM_SIZE * M_UX_INDEX] = ux_t30;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_UY_INDEX] = uy_t30;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_UZ_INDEX] = uz_t30;

    fMom[baseIdx + BLOCK_LBM_SIZE * M_MXX_INDEX] = m_xx_t45;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_MXY_INDEX] = m_xy_t90;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_MXZ_INDEX] = m_xz_t90;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_MYY_INDEX] = m_yy_t45;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_MYZ_INDEX] = m_yz_t90;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_MZZ_INDEX] = m_zz_t45;
    
    #ifdef OMEGA_FIELD
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omegaVar;
    fMom[baseIdx + BLOCK_LBM_SIZE * M_OMEGA_INDEX] = omegaVar;
    #endif


    if(save){
        #ifdef BC_FORCES
        //update local forces
        const int baseIdxScalarBlock = idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z);
        d_BC_Fx[baseIdxScalarBlock] = (L_BC_Fx);
        d_BC_Fy[baseIdxScalarBlock] = (L_BC_Fy);
        d_BC_Fz[baseIdxScalarBlock] = (L_BC_Fz);
        #endif 
    }
    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        #ifdef SECOND_DIST 
            udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - ux_t30);
            udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uy_t30);
            udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uz_t30);

            #include COLREC_G_RECONSTRUCTION

            #include "includeFiles/g_popSave.inc"
            
            fMom[baseIdx + BLOCK_LBM_SIZE * M2_C_INDEX] = cVar;
            fMom[baseIdx + BLOCK_LBM_SIZE * M2_CX_INDEX] = qx_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * M2_CY_INDEX] = qy_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * M2_CZ_INDEX] = qz_t30;

        #endif
        #ifdef A_XX_DIST
            Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
            Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
            Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

            #include COLREC_AXX_RECONSTRUCTION

            #include "includeFiles/conformationTransport\popSave_Axx.inc"
           
           fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_C_INDEX] = AxxVar;
           fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_CX_INDEX] = Axx_qx_t30;
           fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_CY_INDEX] = Axx_qy_t30;
           fMom[baseIdx + BLOCK_LBM_SIZE * A_XX_CZ_INDEX] = Axx_qz_t30;
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
            Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
            Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
            Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

            #include COLREC_AXY_RECONSTRUCTION

            #include "includeFiles/conformationTransport\popSave_Axy.inc"
           
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_C_INDEX] = AxyVar;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_CX_INDEX] = Axy_qx_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_CY_INDEX] = Axy_qy_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XY_CZ_INDEX] = Axy_qz_t30;
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
            Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
            Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
            Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

            #include COLREC_AXZ_RECONSTRUCTION

            #include "includeFiles/conformationTransport\popSave_Axz.inc"
           
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_C_INDEX] = AxzVar;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_CX_INDEX] = Axz_qx_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_CY_INDEX] = Axz_qy_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_XZ_CZ_INDEX] = Axz_qz_t30;
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
            Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
            Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
            Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

            #include COLREC_AYY_RECONSTRUCTION

            #include "includeFiles/conformationTransport\popSave_Ayy.inc"
           
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_C_INDEX] = AyyVar;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_CX_INDEX] = Ayy_qx_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_CY_INDEX] = Ayy_qy_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YY_CZ_INDEX] = Ayy_qz_t30;
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
            Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
            Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
            Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

            #include COLREC_AYZ_RECONSTRUCTION

            #include "includeFiles/conformationTransport\popSave_Ayz.inc"
           
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_C_INDEX] = AyzVar;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_CX_INDEX] = Ayz_qx_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_CY_INDEX] = Ayz_qy_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_YZ_CZ_INDEX] = Ayz_qz_t30;
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
            Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
            Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
            Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

            #include COLREC_AZZ_RECONSTRUCTION

            #include "includeFiles/conformationTransport\popSave_Azz.inc"

            fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_C_INDEX] = AzzVar;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_CX_INDEX] = Azz_qx_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_CY_INDEX] = Azz_qy_t30;
            fMom[baseIdx + BLOCK_LBM_SIZE * A_ZZ_CZ_INDEX] = Azz_qz_t30;
        #endif //A_ZZ_DIST
    #endif //CONVECTION_DIFFUSION_TRANSPORT

    #include "includeFiles/popSave.inc"

    //save velocities in the end in order to load next step to compute the gradient
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
    //#include "includeFiles/velSave.inc"
    //save conformation tensor components in the halo
    //#include "includeFIles/conformationTransport/confSave.inc"
    #endif //COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

}
