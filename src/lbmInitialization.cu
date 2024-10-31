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
    dfloat *fMom, float* randomNumbers)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);

    //first moments
    dfloat rho, ux, uy, uz;
    #ifdef NON_NEWTONIAN_FLUID
    dfloat omega;
    #endif
    #ifdef SECOND_DIST 
    dfloat cVar, qx_t30, qy_t30, qz_t30;
    #endif
    
    #include CASE_FLOW_INITIALIZATION

    #ifdef NON_NEWTONIAN_FLUID
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

    #ifdef NON_NEWTONIAN_FLUID
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omega;
    #endif   
    #ifdef SECOND_DIST 


    dfloat invC= 1.0/cVar;

    //TODO: fix initialization when a flux exist, since i initialize with zero, there is no problem currently
    dfloat udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC*0.0 - ux*F_M_I_SCALE);
    dfloat udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC*0.0 - uy*F_M_I_SCALE);
    dfloat udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC*0.0 - uz*F_M_I_SCALE);

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = cVar;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qx_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qz_t30;

    #endif 
}

__global__ void gpuInitialization_pop(
    dfloat *fMom,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1
    #ifdef SECOND_DIST 
    ,dfloat *g_fGhostX_0, dfloat *g_fGhostX_1,
    dfloat *g_fGhostY_0, dfloat *g_fGhostY_1,
    dfloat *g_fGhostZ_0, dfloat *g_fGhostZ_1
    #endif 
    )
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
    #include COLREC_RECONSTRUCTIONS
    
    //thread xyz
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

    #ifdef SECOND_DIST 
        dfloat gNode[GQ];
        
        dfloat cVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invC = 1/cVar;
        dfloat qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - ux_t30);
        dfloat udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uy_t30);
        dfloat udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uz_t30);

        #include COLREC_G_RECONSTRUCTIONS

        if (threadIdx.x == 0) { //w
            g_fGhostX_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 2]; 
            #ifdef D3G19
            g_fGhostX_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 8];
            g_fGhostX_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[10];
            g_fGhostX_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[14];
            g_fGhostX_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[16];
            #endif            
        }else if (threadIdx.x == (BLOCK_NX - 1)){                    
            g_fGhostX_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = gNode[ 1];
            #ifdef D3G19
            g_fGhostX_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = gNode[ 7];
            g_fGhostX_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = gNode[ 9];
            g_fGhostX_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = gNode[13];
            g_fGhostX_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = gNode[15];     
            #endif    
        }

        if (threadIdx.y == 0)  { //s                             
            g_fGhostY_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 4];
            #ifdef D3G19
            g_fGhostY_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 8];
            g_fGhostY_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[12];
            g_fGhostY_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[13];
            g_fGhostY_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[18];           
            #endif           
        }else if (threadIdx.y == (BLOCK_NY - 1)){             
            g_fGhostY_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = gNode[ 3];
            #ifdef D3G19
            g_fGhostY_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = gNode[ 7];
            g_fGhostY_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = gNode[11];
            g_fGhostY_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = gNode[14];
            g_fGhostY_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = gNode[17];         
            #endif        
        }
        
        if (threadIdx.z == 0){ //b                          
            g_fGhostZ_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 6];
            #ifdef D3G19
            g_fGhostZ_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[10];
            g_fGhostZ_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[12];
            g_fGhostZ_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[15];
            g_fGhostZ_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[17]; 
            #endif    
        }else if (threadIdx.z == (BLOCK_NZ - 1)){                  
            g_fGhostZ_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = gNode[ 5];
            #ifdef D3G19
            g_fGhostZ_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = gNode[ 9];
            g_fGhostZ_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = gNode[11];
            g_fGhostZ_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = gNode[16];
            g_fGhostZ_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = gNode[18];    
            #endif                    
        }
    #endif //SECOND_DIST


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


__host__ void hostInitialization_nodeType_bulk(
    unsigned int *hNodeType)
{
    int x,y,z;
    unsigned int nodeType;

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

    printf("boundary condition done\n");
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
    int value;
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
        //printf("x %d y %d z %d \n",x,y,z); fflush(stdout);
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

/*
void define_voxel_bc(
    unsigned int *dNodeType
){
    for(int x= 0;x<NX;x++){
        for(int y =0; y<NY;y++){
            for(int z =0; z<NZ_TOTAL;z++){
                unsigned int index = idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ);
                if(dNodeType[index] == MISSING_DEFINITION){
                    dNodeType[index] = bc_id(dNodeType,x,y,z);
                }
            }
        }
    }
}
*/

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