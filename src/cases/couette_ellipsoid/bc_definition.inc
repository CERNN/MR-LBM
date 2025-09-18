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
case SOUTH:
    ux_t30 = -U_MAX;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = (6*U_MAX*U_MAX*rho_I)/5;
    m_xy_t90 = (5*m_xy_I)/3 - U_MAX/3;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH:
    ux_t30 = U_MAX;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = (6*U_MAX*U_MAX*rho_I)/5;
    m_xy_t90 = (5*m_xy_I)/3 - U_MAX/3;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
}