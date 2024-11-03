dfloat rho_I;
dfloat inv_rho_I;

dfloat m_xx_I;
dfloat m_xy_I;
dfloat m_xz_I;
dfloat m_yy_I;
dfloat m_yz_I;
dfloat m_zz_I;

dfloat rho;
dfloat inv_rho;

// Check if solid body walls
//if ((nodeType & ((1 << 8) | (1 << 9) | (1 << 10))) == 0) {
 
//#include "voxel"


switch (nodeType){ //z = NZ-1
    case  (FRONT + BC_OUTFLOW):
      

        rhoVar = RHO_0 + fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        ux_t30     =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_UX_INDEX , blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_I_SCALE;
        uy_t30     =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_UY_INDEX , blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_I_SCALE;
        uz_t30     =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_UZ_INDEX , blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_I_SCALE;
        m_xx_t45   =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_II_SCALE;
        m_xy_t90   =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_IJ_SCALE;
        m_xz_t90   =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_IJ_SCALE;
        m_yy_t45   =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_II_SCALE;
        m_yz_t90   =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_IJ_SCALE;
        m_zz_t45   =     fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]/F_M_II_SCALE;


        
        break;
    case  (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0):
   
        
        ux_t30 = 0;
        uy_t30 = 0;
        uz_t30 = U_MAX;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
        inv_rho_I = 1.0 / rho_I;
        m_xz_I = inv_rho_I * (pop[9] - pop[16]);
        m_yz_I = inv_rho_I * (pop[11] - pop[18]);

        rho = RHO_0;

        m_xx_t45 = 0;
        m_xy_t90 = 0;
        m_xz_t90 = (2*m_xz_I*rho_I);
        m_yy_t45 = 0;
        m_yz_t90 = (2*m_yz_I*rho_I);
        m_zz_t45 = uz_t30*uz_t30*rho;

        rhoVar = rho;

    break;
    case (BACK):
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    rho = (3*rho_I*(3*m_zz_I - 3*omegaVar*m_zz_I + 4))/(omegaVar + 9);
    

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_yy_I))/(99*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_xy_t90 = (m_xy_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_xz_t90 = (2*m_xz_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_yy_t45 = -(4*(m_xx_I - 10*m_yy_I)*(omegaVar + 9))/(99*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));
    m_zz_t45 = (15*m_zz_I + 2)/(3*(3*m_zz_I - 3*omegaVar*m_zz_I + 4));

    rhoVar = rho;
    break;
        /*
        case (EAST + BC_FREESLIP):
        break;
        case (WEST + BC_FREESLIP):
        break;
        case (NORTH + BC_FREESLIP):
        break;
        case (SOUTH + BC_FREESLIP):
        break;*/
}
 