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
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    rho = (3*rho_I*(3*m_yy_I - 3*omegaVar*m_yy_I + 4))/(omegaVar + 9);
    inv_rho = 1.0 / rho;

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_zz_I))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xy_t90 = (2*m_xy_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xz_t90 = (m_xz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yy_t45 = (15*m_yy_I + 2)/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_zz_t45 = -(4*(m_xx_I - 10*m_zz_I)*(omegaVar + 9))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));

    rhoVar = rho;

    break;
case NORTH:
    ux_t30 = U_MAX;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    rho = (3*rho_I*(3*m_yy_I - 3*omegaVar*m_yy_I + 4))/(omegaVar + 9);
    

    m_xx_t45 = (4*(omegaVar + 9)*(10*m_xx_I - m_zz_I))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xy_t90 = (18*m_xy_I - 4*U_MAX + 2*omegaVar*m_xy_I - 3*U_MAX*m_yy_I + 3*omegaVar*U_MAX*m_yy_I)/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_xz_t90 = (m_xz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yy_t45 = (15*m_yy_I + 2)/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_yz_t90 = (2*m_yz_I*(omegaVar + 9))/(3*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));
    m_zz_t45 = -(4*(m_xx_I - 10*m_zz_I)*(omegaVar + 9))/(99*(3*m_yy_I - 3*omegaVar*m_yy_I + 4));

    rhoVar = rho;

    break;
}