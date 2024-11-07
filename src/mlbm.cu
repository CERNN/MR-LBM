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
    #ifdef SECOND_DIST
    dfloat gNode[GQ];
    #endif
    dfloat pics2;
    dfloat multiplyTerm;

    #ifdef DYNAMIC_SHARED_MEMORY
    extern __shared__ dfloat s_pop[]; 
    #else
    __shared__ dfloat s_pop[BLOCK_LBM_SIZE * (Q - 1)];
    #endif


    // Load moments from global memory

    //rho'
    unsigned int nodeType = dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)];
    if (nodeType == 0b11111111)  return;
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

    #ifdef NON_NEWTONIAN_FLUID
        dfloat omegaVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
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
    dfloat L_Fx = FX;
    dfloat L_Fy = FY;
    dfloat L_Fz = FZ;

    #ifdef BC_FORCES
    dfloat L_BC_Fx = 0.0;
    dfloat L_BC_Fy = 0.0;
    dfloat L_BC_Fz = 0.0;
    #endif


    #include COLREC_RECONSTRUCTIONS

    const unsigned short int xp1 = (threadIdx.x + 1 + BLOCK_NX) % BLOCK_NX;
    const unsigned short int xm1 = (threadIdx.x - 1 + BLOCK_NX) % BLOCK_NX;

    const unsigned short int yp1 = (threadIdx.y + 1 + BLOCK_NY) % BLOCK_NY;
    const unsigned short int ym1 = (threadIdx.y - 1 + BLOCK_NY) % BLOCK_NY;

    const unsigned short int zp1 = (threadIdx.z + 1 + BLOCK_NZ) % BLOCK_NZ;
    const unsigned short int zm1 = (threadIdx.z - 1 + BLOCK_NZ) % BLOCK_NZ;
    
    //need to compute the gradient before they are streamed
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

    //define shared memory
    __shared__ dfloat sharedVel[VEL_GRAD_BLOCK_SIZE];

    sharedVel[idxVelBlock(tx, ty, tz, 0)] = ux_t30;
    sharedVel[idxVelBlock(tx, ty, tz, 1)] = uy_t30;
    sharedVel[idxVelBlock(tx, ty, tz, 2)] = uz_t30;

    // Load neighboring halo cells in shared memory
    // TODO: will have to create storage for the moments similar to the populations, since the velocities will change for each thread.
    if (tx == 0) {
        sharedVel[idxVelBlock(tx - 1, ty, tz, 0)] = fMom[idxMom(txm1, ty, tz, M_UX_INDEX, bxm1, by, bz)];
        sharedVel[idxVelBlock(tx - 1, ty, tz, 1)] = fMom[idxMom(txm1, ty, tz, M_UY_INDEX, bxm1, by, bz)];
        sharedVel[idxVelBlock(tx - 1, ty, tz, 2)] = fMom[idxMom(txm1, ty, tz, M_UZ_INDEX, bxm1, by, bz)];
    }
    if (tx == BLOCK_NX - 1) {
        sharedVel[idxVelBlock(tx + 1, ty, tz, 0)] = fMom[idxMom(txp1, ty, tz, M_UX_INDEX, bxp1, by, bz)];
        sharedVel[idxVelBlock(tx + 1, ty, tz, 1)] = fMom[idxMom(txp1, ty, tz, M_UY_INDEX, bxp1, by, bz)];
        sharedVel[idxVelBlock(tx + 1, ty, tz, 2)] = fMom[idxMom(txp1, ty, tz, M_UZ_INDEX, bxp1, by, bz)];
    }
    if (ty == 0) {
        sharedVel[idxVelBlock(tx, ty - 1, tz, 0)] = fMom[idxMom(tx, tym1, tz, M_UX_INDEX, bx, bym1, bz)];
        sharedVel[idxVelBlock(tx, ty - 1, tz, 1)] = fMom[idxMom(tx, tym1, tz, M_UY_INDEX, bx, bym1, bz)];
        sharedVel[idxVelBlock(tx, ty - 1, tz, 2)] = fMom[idxMom(tx, tym1, tz, M_UZ_INDEX, bx, bym1, bz)];
    }
    if (ty == BLOCK_NY - 1) {
        sharedVel[idxVelBlock(tx, ty + 1, tz, 0)] = fMom[idxMom(tx, typ1, tz, M_UX_INDEX, bx, byp1, bz)];
        sharedVel[idxVelBlock(tx, ty + 1, tz, 1)] = fMom[idxMom(tx, typ1, tz, M_UY_INDEX, bx, byp1, bz)];
        sharedVel[idxVelBlock(tx, ty + 1, tz, 2)] = fMom[idxMom(tx, typ1, tz, M_UZ_INDEX, bx, byp1, bz)];
    }
    if (tz == 0) {
        sharedVel[idxVelBlock(tx, ty, tz - 1, 0)] = fMom[idxMom(tx, ty, tzm1, M_UX_INDEX, bx, by, bzm1)];
        sharedVel[idxVelBlock(tx, ty, tz - 1, 1)] = fMom[idxMom(tx, ty, tzm1, M_UY_INDEX, bx, by, bzm1)];
        sharedVel[idxVelBlock(tx, ty, tz - 1, 2)] = fMom[idxMom(tx, ty, tzm1, M_UZ_INDEX, bx, by, bzm1)];
    }
    if (tz == BLOCK_NZ - 1) {
        sharedVel[idxVelBlock(tx, ty, tz + 1, 0)] = fMom[idxMom(tx, ty, tzp1, M_UX_INDEX, bx, by, bzp1)];
        sharedVel[idxVelBlock(tx, ty, tz + 1, 1)] = fMom[idxMom(tx, ty, tzp1, M_UY_INDEX, bx, by, bzp1)];
        sharedVel[idxVelBlock(tx, ty, tz + 1, 2)] = fMom[idxMom(tx, ty, tzp1, M_UZ_INDEX, bx, by, bzp1)];
    }
    
    __syncthreads();


    // Compute unrolled gradient calculations
    dfloat duxdx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 0)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 0)]) / 2;
    dfloat duxdy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 0)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 0)]) / 2;
    dfloat duxdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 0)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 0)]) / 2;

    dfloat duydx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 1)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 1)]) / 2;
    dfloat duydy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 1)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 1)]) / 2;
    dfloat duydz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 1)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 1)]) / 2;

    dfloat duzdx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 2)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 2)]) / 2;
    dfloat duzdy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 2)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 2)]) / 2;
    dfloat duzdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 2)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 2)]) / 2;

    #endif //COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE

    
    //save populations in shared memory
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = pop[ 1];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = pop[ 2];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = pop[ 3];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = pop[ 4];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = pop[ 5];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = pop[ 6];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = pop[ 7];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = pop[ 8];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = pop[ 9];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = pop[10];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = pop[11];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = pop[12];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = pop[13];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = pop[14];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = pop[15];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = pop[16];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = pop[17];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = pop[18];
    #ifdef D3Q27
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18)] = pop[19];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19)] = pop[20];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20)] = pop[21];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21)] = pop[22];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22)] = pop[23];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23)] = pop[24];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24)] = pop[25];
    s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25)] = pop[26];
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
            ux_t30 = ((pop[1] - pop[2] + pop[7] - pop[ 8] + pop[ 9] - pop[10] + pop[13] - pop[14] + pop[15] - pop[16]) + L_Fx/2) * invRho;
            uy_t30 = ((pop[3] - pop[4] + pop[7] - pop[ 8] + pop[11] - pop[12] + pop[14] - pop[13] + pop[17] - pop[18]) + L_Fy/2) * invRho;
            uz_t30 = ((pop[5] - pop[6] + pop[9] - pop[10] + pop[11] - pop[12] + pop[16] - pop[15] + pop[18] - pop[17]) + L_Fz/2) * invRho;

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
            ux_t30 = ((pop[1] + pop[7] + pop[9] + pop[13] + pop[15] + pop[19] + pop[21] + pop[23] + pop[26])  - (pop[ 2] + pop[ 8] + pop[10] + pop[14] + pop[16] + pop[20] + pop[22] + pop[24] + pop[25]) + 0.5 * L_Fx) * invRho;
            uy_t30 = ((pop[3] + pop[7] + pop[11] + pop[14] + pop[17] + pop[19] + pop[21] + pop[24] + pop[25]) - (pop[ 4] + pop[ 8] + pop[12] + pop[13] + pop[18] + pop[20] + pop[22] + pop[23] + pop[26]) + 0.5 * L_Fy) * invRho;
            uz_t30 = ((pop[5] + pop[9] + pop[11] + pop[16] + pop[18] + pop[19] + pop[22] + pop[23] + pop[25]) - (pop[ 6] + pop[10] + pop[12] + pop[15] + pop[17] + pop[20] + pop[21] + pop[24] + pop[26]) + 0.5 * L_Fz) * invRho;

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

    #ifdef SECOND_DIST 
        dfloat cVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat invC = 1/cVar;
        dfloat qx_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        dfloat qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

        dfloat udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - ux_t30);
        dfloat udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uy_t30);
        dfloat udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uz_t30);

        #include  COLREC_G_RECONSTRUCTIONS

        __syncthreads();

        //overwrite values
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = gNode[ 1];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = gNode[ 2];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = gNode[ 3];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = gNode[ 4];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = gNode[ 5];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = gNode[ 6];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = gNode[ 7];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = gNode[ 8];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = gNode[ 9];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = gNode[10];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = gNode[11];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = gNode[12];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = gNode[13];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = gNode[14];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = gNode[15];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = gNode[16];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = gNode[17];
        s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = gNode[18];

        
        //sync threads of the block so all populations are saved
        __syncthreads();

        /* pull */
        gNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
        gNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
        gNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
        gNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
        gNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
        gNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
        gNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
        gNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
        gNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
        gNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
        gNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
        gNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
        gNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
        gNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
        gNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
        gNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
        gNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
        gNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];
                
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

        udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - ux_t30);
        udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uy_t30);
        udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uz_t30);
            

        #include COLREC_G_RECONSTRUCTIONS

        #include "includeFiles/g_popSave.inc"
        
        fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = cVar;
        fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qx_t30;
        fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qy_t30;
        fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qz_t30;
    #endif


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
        

        // MOMENTS DETERMINED, COMPUTE OMEGA IF NON-NEWTONIAN FLUID
        #if defined(NON_NEWTONIAN_FLUID) || defined(LES_MODEL)
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
                #ifdef NON_NEWTONIAN_FLUID
                    /*
                    dfloat eta = (1.0/omegaVar - 0.5) / 3.0;
                    dfloat gamma_dot = (1 - 0.5 * (omegaVar)) * auxStressMag / eta;
                    eta = VISC + S_Y/gamma_dot;
                    omegaVar = omegaVar;// 1.0 / (0.5 + 3.0 * eta);
                    */
                if(step> NNF_TRIGGER_STEP){
                    omegaVar = calcOmega(omegaVar, auxStressMag,step);
                }else{
                    omegaVar = OMEGA;
                }


                #endif//  NON_NEWTONIAN_FLUID

                #ifdef LES_MODEL
                    dfloat tau_t = 0.5*sqrt(TAU*TAU+Implicit_const*auxStressMag)-0.5*TAU;
                    dfloat visc_turb_var = tau_t/3.0;

                    omegaVar = 1.0/(TAU + tau_t);
                #endif
                t_omegaVar = 1 - omegaVar;
                tt_omegaVar = 1 - 0.5*omegaVar;
                omegaVar_d2 = omegaVar / 2.0;
                tt_omega_t3 = tt_omegaVar * 3.0;
        #endif 
        
            // COLLIDE
        #include COLREC_COLLISION
        

    //calculate post collision populations
    #include COLREC_RECONSTRUCTIONS
    
    
    /* write to global mom */

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = rhoVar - RHO_0;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = ux_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = uy_t30;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = uz_t30;

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xx_t45;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xy_t90;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_xz_t90;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yy_t45;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_yz_t90;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = m_zz_t45;
    
    #ifdef NON_NEWTONIAN_FLUID
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omegaVar;
    #endif

    if(save){
        #ifdef BC_FORCES
        //update local forces
        d_BC_Fx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_BC_Fx);
        d_BC_Fy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_BC_Fy);
        d_BC_Fz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = (L_BC_Fz);
        #endif 
    }


    #include "includeFiles/popSave.inc"
}
