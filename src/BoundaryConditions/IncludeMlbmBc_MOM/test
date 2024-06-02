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
 
switch (nodeType){
    case  (FRONT + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0):
      
        ux_t30 = 0;
        uy_t30 = 0;
        uz_t30 = U_MAX;

        rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
        inv_rho_I = 1.0 / rho_I;
        m_xz_I = inv_rho_I * (pop[10] - pop[15]);
        m_yz_I = inv_rho_I * (pop[12] - pop[17]);

        rho = RHO_0;

        m_xx_t45 = 0;
        m_xy_t90 = 0;
        m_xz_t90 = (2*m_xz_I*rho_I);
        m_yy_t45 = 0;
        m_yz_t90 = (2*m_yz_I*rho_I);
        m_zz_t45 = uz_t30*uz_t30*rho;

        rhoVar = rho;



        
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
 