#include "lbmInitialization.cuh"



__host__
void initializationRandomNumbers(
    float* randomNumbers, int seed)
{
    curandGenerator_t gen;

    // Create pseudo-random number generator
    checkCurandStatus(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT));
    
    // Set generator seed
    checkCurandStatus(curandSetPseudoRandomGeneratorSeed(gen,
        CURAND_SEED));
    
    // Generate NX*NY*NZ floats on device, using normal distribution
    // with mean=0 and std_dev=NORMAL_STD_DEV
    checkCurandStatus(curandGenerateNormal(gen, randomNumbers, NUMBER_LBM_NODES,
        0, CURAND_STD_DEV));

    checkCurandStatus(curandDestroyGenerator(gen));
}


__global__ void gpuInitialization_mom(
    dfloat *fMom, dfloat* randomNumbers)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);

    //first moments
    dfloat rho = RHO_0, ux = U_0_X, uy = U_0_Y, uz = U_0_Z;
    #ifdef OMEGA_FIELD
    dfloat omega;
    #endif
    #ifdef SECOND_DIST 
    dfloat cVar = 1.0;
    dfloat qx_t30 = 3.0*cVar*(ux - 0.0);
    dfloat qy_t30 = 3.0*cVar*(uy - 0.0);
    dfloat qz_t30 = 3.0*cVar*(uz - 0.0);
    #endif
    #ifdef  CONFORMATION_TENSOR
        //assuming that velocity has grad = 0 
        #ifdef A_XX_DIST 
        dfloat AxxVar = 1.0 + CONF_ZERO; 
        dfloat Axx_qx_t30 = 3.0*AxxVar*(ux + 0.0);
        dfloat Axx_qy_t30 = 3.0*AxxVar*(uy + 0.0);
        dfloat Axx_qz_t30 = 3.0*AxxVar*(uz + 0.0);
        #endif
        #ifdef A_XY_DIST 
        dfloat AxyVar = 1.0 + CONF_ZERO;
        dfloat Axy_qx_t30 = 3.0*AxyVar*(ux + 0.0);
        dfloat Axy_qy_t30 = 3.0*AxyVar*(uy + 0.0);
        dfloat Axy_qz_t30 = 3.0*AxyVar*(uz + 0.0);
        #endif
        #ifdef A_XZ_DIST 
        dfloat AxzVar = 1.0 + CONF_ZERO;
        dfloat Axz_qx_t30 = 3.0*AxzVar*(ux + 0.0);
        dfloat Axz_qy_t30 = 3.0*AxzVar*(uy + 0.0);
        dfloat Axz_qz_t30 = 3.0*AxzVar*(uz + 0.0);
        #endif
        #ifdef A_YY_DIST 
        dfloat AyyVar = 1.0 + CONF_ZERO;
        dfloat Ayy_qx_t30 = 3.0*AyyVar*(ux + 0.0);
        dfloat Ayy_qy_t30 = 3.0*AyyVar*(uy + 0.0);
        dfloat Ayy_qz_t30 = 3.0*AyyVar*(uz + 0.0);
        #endif
        #ifdef A_YZ_DIST 
        dfloat AyzVar = 1.0 + CONF_ZERO;
        dfloat Ayz_qx_t30 = 3.0*AyzVar*(ux + 0.0);
        dfloat Ayz_qy_t30 = 3.0*AyzVar*(uy + 0.0);
        dfloat Ayz_qz_t30 = 3.0*AyzVar*(uz + 0.0);
        #endif
        #ifdef A_ZZ_DIST 
        dfloat AzzVar = 1.0 + CONF_ZERO;
        dfloat Azz_qx_t30 = 3.0*AzzVar*(ux + 0.0);
        dfloat Azz_qy_t30 = 3.0*AzzVar*(uy + 0.0);
        dfloat Azz_qz_t30 = 3.0*AzzVar*(uz + 0.0);
        #endif
    #endif


    #include CASE_FLOW_INITIALIZATION

    #ifdef OMEGA_FIELD
    omega = OMEGA;
    #endif

   
    // zeroth moment
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = rho-RHO_0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE*ux;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE*uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_I_SCALE*uz;

    //second moments
    //define equilibrium populations
    dfloat pop[Q];
    for (int i = 0; i < Q; i++)
    {
        pop[i] = gpu_f_eq(w[i] * RHO_0,
                          3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1 - 1.5 * (ux * ux + uy * uy + uz * uz));
    }
    
    dfloat invRho = 1.0/rho;
    dfloat pixx =  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
    dfloat pixy = ((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho;
    dfloat pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
    dfloat piyy =  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
    dfloat piyz = ((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho;
    dfloat pizz =  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*pixx;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*pixy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*pixz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*piyy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*piyz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*pizz;

    #ifdef OMEGA_FIELD
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omega;
    #endif 
    
    
    #ifdef SECOND_DIST 
    dfloat invC= 1.0/cVar;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = cVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qz_t30;
    #endif 

    #ifdef A_XX_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AxxVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qz_t30;
    #endif 
    #ifdef A_XY_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AxyVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qz_t30;
    #endif 
    #ifdef A_XZ_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AxzVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qz_t30;
    #endif
    #ifdef A_YY_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AyyVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qz_t30;
    #endif
    #ifdef A_YZ_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AyzVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qz_t30;
    #endif
    #ifdef A_ZZ_DIST 
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  AzzVar;
    //fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] =  0.0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qz_t30;
    #endif

    #ifdef LOCAL_FORCES
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FX;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FY;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_FZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = FZ;
    #endif 


}

__global__ void gpuInitialization_pop(
    dfloat *fMom, ghostInterfaceData ghostInterface)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);
    // zeroth moment

    dfloat rhoVar = RHO_0 + fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat ux_t30     = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uy_t30     = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uz_t30     = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xx_t45   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xy_t90   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_xz_t90   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_yy_t45   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_yz_t90   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat m_zz_t45   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat pop[Q];
    dfloat multiplyTerm;
    dfloat pics2;
    #include COLREC_RECONSTRUCTION
    
    //thread xyz
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    
    //block xyz
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    if (threadIdx.x == 0) { //w
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2]; 
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
        #ifdef D3Q27                                                                                                           
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
        ghostInterface.fGhost.X_0[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.x == (BLOCK_NX - 1)){                                                                                                                                                                               
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
        #ifdef D3Q27                                                                                                           
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
        ghostInterface.fGhost.X_1[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27       
    }

    if (threadIdx.y == 0)  { //s                                                                                                                                                                                        
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 4];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 8];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 2, bx, by, bz)] = pop[12];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 3, bx, by, bz)] = pop[13];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
        ghostInterface.fGhost.Y_0[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.y == (BLOCK_NY - 1)){                                                                                                                                                                        
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 3];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 7];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 2, bx, by, bz)] = pop[11];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 3, bx, by, bz)] = pop[14];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
        ghostInterface.fGhost.Y_1[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }
    
    if (threadIdx.z == 0){ //b                                                                                                                                                                                     
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 6];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[10];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[12];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[15];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
        ghostInterface.fGhost.Z_0[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.z == (BLOCK_NZ - 1)){                                                                                                               
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 5];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[ 9];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[11];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[16];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
        ghostInterface.fGhost.Z_1[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                                                                                                                                    
    }

    #ifdef CONVECTION_DIFFUSION_TRANSPORT
        dfloat gNode[GQ];


    #ifdef SECOND_DIST 
        
        dfloat cVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invC = 1/cVar;
        dfloat qx_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qy_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qz_t30 = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - ux_t30);
        dfloat udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uy_t30);
        dfloat udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uz_t30);

        #include COLREC_G_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.g_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.g_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.g_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.g_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.g_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.g_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.g_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.g_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.g_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.g_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.g_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.g_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.g_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.g_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.g_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.g_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.g_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.g_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.g_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.g_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.g_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.g_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.g_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.g_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.g_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.g_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.g_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.g_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.g_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.g_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //SECOND_DIST
    #ifdef A_XX_DIST 
        
        dfloat AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAxx = 1/AxxVar;
        dfloat Axx_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axx_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axx_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat Axx_udx_t30 = CONF_DIFF_FLUC_COEF * (Axx_qx_t30*invAxx - ux_t30);
        dfloat Axx_udy_t30 = CONF_DIFF_FLUC_COEF * (Axx_qy_t30*invAxx - uy_t30);
        dfloat Axx_udz_t30 = CONF_DIFF_FLUC_COEF * (Axx_qz_t30*invAxx - uz_t30);

        #include COLREC_AXX_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        
        dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAxy = 1/AxyVar;
        dfloat Axy_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axy_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axy_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat Axy_udx_t30 = CONF_DIFF_FLUC_COEF * (Axy_qx_t30*invAxy - ux_t30);
        dfloat Axy_udy_t30 = CONF_DIFF_FLUC_COEF * (Axy_qy_t30*invAxy - uy_t30);
        dfloat Axy_udz_t30 = CONF_DIFF_FLUC_COEF * (Axy_qz_t30*invAxy - uz_t30);

        #include COLREC_AXY_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST 
        
        dfloat AxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAxz = 1/AxzVar;
        dfloat Axz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Axz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat Axz_udx_t30 = CONF_DIFF_FLUC_COEF * (Axz_qx_t30*invAxz - ux_t30);
        dfloat Axz_udy_t30 = CONF_DIFF_FLUC_COEF * (Axz_qy_t30*invAxz - uy_t30);
        dfloat Axz_udz_t30 = CONF_DIFF_FLUC_COEF * (Axz_qz_t30*invAxz - uz_t30);

        #include COLREC_AXZ_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST 
        
        dfloat AyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAyy = 1/AyyVar;
        dfloat Ayy_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayy_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayy_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat Ayy_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qx_t30*invAyy - ux_t30);
        dfloat Ayy_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qy_t30*invAyy - uy_t30);
        dfloat Ayy_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayy_qz_t30*invAyy - uz_t30);

        #include COLREC_AYY_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        
        dfloat AyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAyz = 1/AyzVar;
        dfloat Ayz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Ayz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat Ayz_udx_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qx_t30*invAyz - ux_t30);
        dfloat Ayz_udy_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qy_t30*invAyz - uy_t30);
        dfloat Ayz_udz_t30 = CONF_DIFF_FLUC_COEF * (Ayz_qz_t30*invAyz - uz_t30);

        #include COLREC_AYZ_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        
        dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invAzz = 1/AzzVar;
        dfloat Azz_qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Azz_qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat Azz_qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat Azz_udx_t30 = CONF_DIFF_FLUC_COEF * (Azz_qx_t30*invAzz - ux_t30);
        dfloat Azz_udy_t30 = CONF_DIFF_FLUC_COEF * (Azz_qy_t30*invAzz - uy_t30);
        dfloat Azz_udz_t30 = CONF_DIFF_FLUC_COEF * (Azz_qz_t30*invAzz - uz_t30);

        #include COLREC_AZZ_RECONSTRUCTION

        if (threadIdx.x == 0) { //w
            ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //A_ZZ_DIST
    #endif //CONVECTION_DIFFUSION_TRANSPORT
    /*
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

    if (threadIdx.x == 0) { //w
            ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 0, bx, by, bz)] = ux_t30;
            ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 1, bx, by, bz)] = uy_t30;
            ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 2, bx, by, bz)] = uz_t30;
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 0, bx, by, bz)] = ux_t30;
            ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 1, bx, by, bz)] = uy_t30;
            ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 2, bx, by, bz)] = uz_t30;
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 0, bx, by, bz)] = ux_t30;
            ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 1, bx, by, bz)] = uy_t30;
            ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 2, bx, by, bz)] = uz_t30;
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 0, bx, by, bz)] = ux_t30;
            ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 1, bx, by, bz)] = uy_t30;
            ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 2, bx, by, bz)] = uz_t30;
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.f_uGhost.Z_0[g_idxUZ(tx, ty, 0, bx, by, bz)] = ux_t30;
            ghostInterface.f_uGhost.Z_0[g_idxUZ(tx, ty, 1, bx, by, bz)] = uy_t30;
            ghostInterface.f_uGhost.Z_0[g_idxUZ(tx, ty, 2, bx, by, bz)] = uz_t30;
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.f_uGhost.Z_1[g_idxUZ(tx, ty, 0, bx, by, bz)] = ux_t30;
            ghostInterface.f_uGhost.Z_1[g_idxUZ(tx, ty, 1, bx, by, bz)] = uy_t30;
            ghostInterface.f_uGhost.Z_1[g_idxUZ(tx, ty, 2, bx, by, bz)] = uz_t30;
        }
    #endif
    
    #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE

    if (threadIdx.x == 0) { //w
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
            ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
            ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
        }

        if (threadIdx.y == 0)  { //s                             
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
            ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
            ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
        }
        
        if (threadIdx.z == 0){ //b                          
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
            ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
            ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
        }
    #endif
    */
    
}


__global__ void gpuInitialization_nodeType(
    unsigned int *dNodeType)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
    unsigned int nodeType;

    #include CASE_BC_INIT

    dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;
}

__host__ void hostInitialization_nodeType_bulk(
    unsigned int *hNodeType)
{
    int x,y,z;
    //unsigned int nodeType;

    for (x = 0; x<NX;x++){
        for (y = 0; y<NY;y++){
            for (z = 0; z<NZ_TOTAL;z++){
                hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = BULK;
            }
        }
    }
    printf("bulk done\n");
}

__host__ void hostInitialization_nodeType(
    unsigned int *hNodeType)
{
    int x,y,z;
    unsigned int nodeType;

    for (x = 0; x<NX;x++){
        for (y = 0; y<NY;y++){
            for (z = 0; z<NZ_TOTAL;z++){
                #include CASE_BC_INIT
                if (nodeType != BULK)
                    hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = (unsigned int)nodeType;
            }
        }
    }

    printf("Setting boundary condition completed\n");
}

__global__ void gpuInitialization_force(
    dfloat *d_BC_Fx, dfloat* d_BC_Fy, dfloat* d_BC_Fz)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);

    d_BC_Fx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0;
    d_BC_Fy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0;
    d_BC_Fz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0; 
}

void read_xyz_file(
    const std::string& filename,
    unsigned int* dNodeType
) {
    std::ifstream csv_file(filename);
    if (!csv_file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int x, y, z;
    size_t index, index_n;

    int xi, yi, zi;

    std::string line;
    while (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string field;

        std::getline(ss, field, ',');
        x = std::stoi(field);

        std::getline(ss, field, ',');
        y = std::stoi(field);

        std::getline(ss, field, ',');
        z = std::stoi(field);

        if((x>=NX)||(y>=NY)||(z>=NZ_TOTAL))
            continue;


        index = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
        dNodeType[idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] = SOLID_NODE;


        //set neighborings to be BC
        for (int xn = -1; xn < 2; xn++) {
            for (int yn = -1; yn < 2; yn++) {
                for (int zn = -1; zn < 2; zn++) {

                    xi = (x + xn + NX) % NX;
                    yi = (y + yn + NY) % NY;
                    zi = (z + zn + NZ) % NZ;


                    index_n = idxScalarBlock(xi% BLOCK_NX, yi % BLOCK_NY, zi % BLOCK_NZ, xi / BLOCK_NX, yi / BLOCK_NY, zi / BLOCK_NZ);

                    if ((index_n == index) || dNodeType[index_n] == 255) // check if is the center of the cuboid or if is already a solid node
                        continue;
                    else //set flag to max int 
                        dNodeType[index_n] = MISSING_DEFINITION;
                }
            }
        }
    }
    csv_file.close();
    printf("voxels imported \n");
}


__global__ 
void define_voxel_bc(
    unsigned int *dNodeType
){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    unsigned int index = idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ);
    if(dNodeType[index] == MISSING_DEFINITION){
        dNodeType[index] = bc_id(dNodeType,x,y,z);
    }
}



/*
Note: Due to the way the BC are set up, it possible when setting a solid node to also set the bit flags of neighboring nodes
However if attempt to perform in device, need to pay attention of two solid nodes setting the same flag at same time 
*/
__host__ __device__
unsigned int bc_id(unsigned int *dNodeType, int x, int y, int z){

    unsigned int bc_d = BULK;

    int xp1 = (x+1+NX)%NX;
    int xm1 = (x-1+NX)%NX;
    int yp1 = (y+1+NY)%NY;
    int ym1 = (y-1+NY)%NY;
    int zp1 = (z+1+NZ)%NZ;
    int zm1 = (z-1+NZ)%NZ;

    // 1
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 3);
        bc_d |= (1 << 5);
        bc_d |= (1 << 7);
    }
     // 2
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 2);
        bc_d |= (1 << 4);
        bc_d |= (1 << 6);
    }
    // 3
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 4
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
    }
    // 5
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 6
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
    }
    // 7
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 3);
        bc_d |= (1 << 7);
    }
    // 8
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 4);
    }
    // 9
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 5);
        bc_d |= (1 << 7);
    }
    // 10
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 2);
    }
    // 11
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 12
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
    }
    // 13
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 5);
    }
    // 14
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 6);
    }
    // 15
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 3);
    }
    // 16
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 6);
    }
    // 17
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
    }
    // 18
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
    }
    // 19
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 7);
    }
    // 20
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
    }
    // 21
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 3);
    }
    // 22
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
    }
    // 23
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 5);
    }
    // 24
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
    }
    // 25
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 6);
    }
    // 26
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);   
    }

    return bc_d;
}