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
case SOUTH_WEST_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_WEST_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[8] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_WEST_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_WEST_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_EAST_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_EAST_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_EAST_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[6] + pop[7] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_EAST_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11];
    inv_rho_I = 1.0 / rho_I;

    rho = (12*rho_I)/7;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_WEST:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[8]);

    rho = (36*rho_I*(OMEGA*m_xy_I - m_xy_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = (25*m_xy_I - 1)/(9*(OMEGA*m_xy_I - m_xy_I + 1));
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_WEST:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (-pop[14]);

    rho = (36*rho_I*(m_xy_I - OMEGA*m_xy_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = (25*m_xy_I + 1)/(9*(m_xy_I - OMEGA*m_xy_I + 1));
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_EAST:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (-pop[13]);

    rho = (36*rho_I*(m_xy_I - OMEGA*m_xy_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = (25*m_xy_I + 1)/(9*(m_xy_I - OMEGA*m_xy_I + 1));
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_EAST:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[7]);

    rho = (36*rho_I*(OMEGA*m_xy_I - m_xy_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = (25*m_xy_I - 1)/(9*(OMEGA*m_xy_I - m_xy_I + 1));
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case WEST_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xz_I = inv_rho_I * (pop[10]);

    rho = (36*rho_I*(OMEGA*m_xz_I - m_xz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = (25*m_xz_I - 1)/(9*(OMEGA*m_xz_I - m_xz_I + 1));
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case WEST_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xz_I = inv_rho_I * (-pop[16]);

    rho = (36*rho_I*(m_xz_I - OMEGA*m_xz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = (25*m_xz_I + 1)/(9*(m_xz_I - OMEGA*m_xz_I + 1));
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case EAST_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xz_I = inv_rho_I * (-pop[15]);

    rho = (36*rho_I*(m_xz_I - OMEGA*m_xz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = (25*m_xz_I + 1)/(9*(m_xz_I - OMEGA*m_xz_I + 1));
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case EAST_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xz_I = inv_rho_I * (pop[9]);

    rho = (36*rho_I*(OMEGA*m_xz_I - m_xz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = (25*m_xz_I - 1)/(9*(OMEGA*m_xz_I - m_xz_I + 1));
    m_yy_t45 = 0;
    m_yz_t90 = 0;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_yz_I = inv_rho_I * (pop[12]);

    rho = (36*rho_I*(OMEGA*m_yz_I - m_yz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (25*m_yz_I - 1)/(9*(OMEGA*m_yz_I - m_yz_I + 1));
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case SOUTH_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_yz_I = inv_rho_I * (-pop[18]);

    rho = (36*rho_I*(m_yz_I - OMEGA*m_yz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (25*m_yz_I + 1)/(9*(m_yz_I - OMEGA*m_yz_I + 1));
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_yz_I = inv_rho_I * (-pop[17]);

    rho = (36*rho_I*(m_yz_I - OMEGA*m_yz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (25*m_yz_I + 1)/(9*(m_yz_I - OMEGA*m_yz_I + 1));
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH_FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_yz_I = inv_rho_I * (pop[11]);

    rho = (36*rho_I*(OMEGA*m_yz_I - m_yz_I + 1))/(OMEGA + 24);
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (25*m_yz_I - 1)/(9*(OMEGA*m_yz_I - m_yz_I + 1));
    m_zz_t45 = 0;

    rhoVar = rho;

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
case SOUTH:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = (5*m_xy_I)/3;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case NORTH:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = (5*m_xy_I)/3;
    m_xz_t90 = 0;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case BACK:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = (5*m_xz_I)/3;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
case FRONT:
    ux_t30 = 0;
    uy_t30 = 0;
    uz_t30 = 0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);

    rho = (6*rho_I)/5;
    

    m_xx_t45 = 0;
    m_xy_t90 = 0;
    m_xz_t90 = (5*m_xz_I)/3;
    m_yy_t45 = 0;
    m_yz_t90 = (5*m_yz_I)/3;
    m_zz_t45 = 0;

    rhoVar = rho;

    break;
}