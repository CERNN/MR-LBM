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

switch (nodeType)
{
case BULK:
    break;
case WEST:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = (5*m_xy_I)/3;
    m_xz_t90 = (5*m_xz_I)/3;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case EAST:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = (5*m_xy_I)/3;
    m_xz_t90 = (5*m_xz_I)/3;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
}