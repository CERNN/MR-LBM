/*
    //IO:
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat m_xx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] - cs2*rho_I);
    dfloat m_xy_I = inv_rho_I * ((pop[7] + pop[8]) - (pop[13] + pop[14]));
    dfloat m_xz_I = inv_rho_I * ((pop[9] + pop[10]) - (pop[15] + pop[16])) ;
    dfloat m_yy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] - cs2*rho_I);
    dfloat m_yz_I = inv_rho_I * ((pop[11] + pop[12]) - (pop[17] + pop[18]));
    dfloat m_zz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] - cs2*rho_I);
*/

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
dfloat inv_rho2;

dfloat ux1;
dfloat ux2;
dfloat ux3;

dfloat bE;
dfloat dE;

dfloat l1, l2, l3;

dfloat p_x1_x1;
dfloat p_x2_x2;
dfloat p_x3_x3;
dfloat p_x1_x2;
dfloat p_x1_x3;
dfloat p_x2_x3;

dfloat p_x1_x1_I;
dfloat p_x2_x2_I;
dfloat p_x3_x3_I;
dfloat p_x1_x2_I;
dfloat p_x1_x3_I;
dfloat p_x2_x3_I;

switch (nodeType)
{

case BULK:
        break;
        //####################### EDGE #######################
case SOUTH_WEST:
    l1 = -1;
    l2 = -1;
    l3 = 0;
    // u1 = ux // u2  = uy // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_yy_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_xz_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I - 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I + 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I - 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I - 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (2412*p_x1_x2_I*rho_I - 504*p_x1_x1_I*rho_I - 228*rho_I - 504*p_x2_x2_I*rho_I + 216*p_x3_x3_I*rho_I + 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I + 79*omegaVar*p_x2_x2_I*rho_I + 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux2;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x2_x2;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x1_x3;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case NORTH_WEST:
    l1 = -1;
    l2 = 1;
    l3 = 0;
    // u1 = ux // u2  = uy // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (-pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_yy_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_xz_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I + 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I - 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I + 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I + 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (228*rho_I + 504*p_x1_x1_I*rho_I + 2412*p_x1_x2_I*rho_I + 504*p_x2_x2_I*rho_I - 216*p_x3_x3_I*rho_I - 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I - 79*omegaVar*p_x2_x2_I*rho_I - 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux2;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x2_x2;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x1_x3;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case SOUTH_EAST:
    l1 = 1;
    l2 = -1;
    l3 = 0;
    // u1 = ux // u2  = uy // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (-pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_yy_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_xz_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I + 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I - 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I + 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I + 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (228*rho_I + 504*p_x1_x1_I*rho_I + 2412*p_x1_x2_I*rho_I + 504*p_x2_x2_I*rho_I - 216*p_x3_x3_I*rho_I - 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I - 79*omegaVar*p_x2_x2_I*rho_I - 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux2;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x2_x2;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x1_x3;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case NORTH_EAST:
    l1 = 1;
    l2 = 1;
    l3 = 0;
    // u1 = ux // u2  = uy // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_yy_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_xz_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I - 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I + 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I - 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I - 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (2412*p_x1_x2_I*rho_I - 504*p_x1_x1_I*rho_I - 228*rho_I - 504*p_x2_x2_I*rho_I + 216*p_x3_x3_I*rho_I + 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I + 79*omegaVar*p_x2_x2_I*rho_I + 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux2;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x2_x2;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x1_x3;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case WEST_BACK:
    l1 = -1;
    l2 = -1;
    l3 = 0;
    // u1 = ux // u2  = uz // u3 == uy
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_yy_I;
    p_x1_x2_I = m_xz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I - 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I + 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I - 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I - 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (2412*p_x1_x2_I*rho_I - 504*p_x1_x1_I*rho_I - 228*rho_I - 504*p_x2_x2_I*rho_I + 216*p_x3_x3_I*rho_I + 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I + 79*omegaVar*p_x2_x2_I*rho_I + 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux3;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x3_x3;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x1_x2;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case WEST_FRONT:
    l1 = -1;
    l2 = 1;
    l3 = 0;
    // u1 = ux // u2  = uz // u3 == uy
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (-pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_yy_I;
    p_x1_x2_I = m_xz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I + 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I - 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I + 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I + 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (228*rho_I + 504*p_x1_x1_I*rho_I + 2412*p_x1_x2_I*rho_I + 504*p_x2_x2_I*rho_I - 216*p_x3_x3_I*rho_I - 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I - 79*omegaVar*p_x2_x2_I*rho_I - 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux3;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x3_x3;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x1_x2;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case EAST_BACK:
    l1 = 1;
    l2 = -1;
    l3 = 0;
    // u1 = ux // u2  = uz // u3 == uy
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (-pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_yy_I;
    p_x1_x2_I = m_xz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I + 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I - 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I + 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I + 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (228*rho_I + 504*p_x1_x1_I*rho_I + 2412*p_x1_x2_I*rho_I + 504*p_x2_x2_I*rho_I - 216*p_x3_x3_I*rho_I - 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I - 79*omegaVar*p_x2_x2_I*rho_I - 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux3;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x3_x3;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x1_x2;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case EAST_FRONT:
    l1 = 1;
    l2 = 1;
    l3 = 0;
    // u1 = ux // u2  = uz // u3 == uy
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11] + pop[13] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_yy_I;
    p_x1_x2_I = m_xz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_yz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I - 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I + 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I - 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I - 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (2412*p_x1_x2_I*rho_I - 504*p_x1_x1_I*rho_I - 228*rho_I - 504*p_x2_x2_I*rho_I + 216*p_x3_x3_I*rho_I + 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I + 79*omegaVar*p_x2_x2_I*rho_I + 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux3;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x3_x3;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x1_x2;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case SOUTH_BACK:
    l1 = -1;
    l2 = -1;
    l3 = 0;
    // u1 = uy // u2  = uz // u3 == ux
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15]);
    m_yz_I = inv_rho_I * (pop[12]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15]);

    p_x1_x1_I = m_yy_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_xx_I;
    p_x1_x2_I = m_yz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_xz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I - 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I + 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I - 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I - 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (2412*p_x1_x2_I*rho_I - 504*p_x1_x1_I*rho_I - 228*rho_I - 504*p_x2_x2_I*rho_I + 216*p_x3_x3_I*rho_I + 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I + 79*omegaVar*p_x2_x2_I*rho_I + 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux3;
    uy_t30 = ux1;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x3_x3;
    m_yy_t45 = inv_rho * p_x1_x1;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x2_x3;
    m_yz_t90 = inv_rho * p_x1_x2;

    rhoVar = rho;

    break;
case SOUTH_FRONT:
    l1 = -1;
    l2 = 1;
    l3 = 0;
    // u1 = uy // u2  = uz // u3 == ux
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (-pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_yy_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_xx_I;
    p_x1_x2_I = m_yz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_xz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I + 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I - 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I + 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I + 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (228*rho_I + 504*p_x1_x1_I*rho_I + 2412*p_x1_x2_I*rho_I + 504*p_x2_x2_I*rho_I - 216*p_x3_x3_I*rho_I - 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I - 79*omegaVar*p_x2_x2_I*rho_I - 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux3;
    uy_t30 = ux1;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x3_x3;
    m_yy_t45 = inv_rho * p_x1_x1;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x2_x3;
    m_yz_t90 = inv_rho * p_x1_x2;

    rhoVar = rho;

    break;
case NORTH_BACK:
    l1 = 1;
    l2 = -1;
    l3 = 0;
    // u1 = uy // u2  = uz // u3 == ux
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (-pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_yy_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_xx_I;
    p_x1_x2_I = m_yz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_xz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I + 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I - 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I + 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I - 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I + 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I + 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (228*rho_I + 504*p_x1_x1_I*rho_I + 2412*p_x1_x2_I*rho_I + 504*p_x2_x2_I*rho_I - 216*p_x3_x3_I*rho_I - 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I - 79*omegaVar*p_x2_x2_I*rho_I - 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux3;
    uy_t30 = ux1;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x3_x3;
    m_yy_t45 = inv_rho * p_x1_x1;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x2_x3;
    m_yz_t90 = inv_rho * p_x1_x2;

    rhoVar = rho;

    break;
case NORTH_FRONT:
    l1 = 1;
    l2 = 1;
    l3 = 0;
    // u1 = uy // u2  = uz // u3 == ux
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16]);
    m_yz_I = inv_rho_I * (pop[11]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16]);

    p_x1_x1_I = m_yy_I;
    p_x2_x2_I = m_zz_I;
    p_x3_x3_I = m_xx_I;
    p_x1_x2_I = m_yz_I;
    p_x1_x3_I = m_xy_I;
    p_x2_x3_I = m_xz_I;

    rho = (36*(23*rho_I + 24*p_x1_x1_I*rho_I - 57*p_x1_x2_I*rho_I + 24*p_x2_x2_I*rho_I - 6*p_x3_x3_I*rho_I - 24*omegaVar*p_x1_x1_I*rho_I + 57*omegaVar*p_x1_x2_I*rho_I - 24*omegaVar*p_x2_x2_I*rho_I + 6*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (192*rho_I + 936*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 216*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I + 239*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I - 191*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x2_x2 = (192*rho_I + 216*p_x1_x1_I*rho_I - 1008*p_x1_x2_I*rho_I + 936*p_x2_x2_I*rho_I - 144*p_x3_x3_I*rho_I - 191*omegaVar*p_x1_x1_I*rho_I + 158*omegaVar*p_x1_x2_I*rho_I + 239*omegaVar*p_x2_x2_I*rho_I - 6*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x3_x3 = -(2*(24*rho_I + 72*p_x1_x1_I*rho_I - 216*p_x1_x2_I*rho_I + 72*p_x2_x2_I*rho_I - 288*p_x3_x3_I*rho_I + 3*omegaVar*p_x1_x1_I*rho_I - 34*omegaVar*p_x1_x2_I*rho_I + 3*omegaVar*p_x2_x2_I*rho_I - 162*omegaVar*p_x3_x3_I*rho_I))/(5*(43*omegaVar + 72));
    p_x1_x2 = (2412*p_x1_x2_I*rho_I - 504*p_x1_x1_I*rho_I - 228*rho_I - 504*p_x2_x2_I*rho_I + 216*p_x3_x3_I*rho_I + 79*omegaVar*p_x1_x1_I*rho_I + 538*omegaVar*p_x1_x2_I*rho_I + 79*omegaVar*p_x2_x2_I*rho_I + 34*omegaVar*p_x3_x3_I*rho_I)/(5*(43*omegaVar + 72));
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = 2*p_x2_x3_I*rho_I;

    ux_t30 = ux3;
    uy_t30 = ux1;
    uz_t30 = ux2;

    m_xx_t45 = inv_rho * p_x3_x3;
    m_yy_t45 = inv_rho * p_x1_x1;
    m_zz_t45 = inv_rho * p_x2_x2;
    m_xy_t90 = inv_rho * p_x1_x3;
    m_xz_t90 = inv_rho * p_x2_x3;
    m_yz_t90 = inv_rho * p_x1_x2;

    rhoVar = rho;

    break;


        //####################### FACE #######################
case WEST:
    l1 = -1;
    l2 = 0;
    l3 = 0;
    // u1 = ux // u2  = uy // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12] + pop[14] + pop[16] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_yy_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_xz_I;
    p_x2_x3_I = m_yz_I;

    rho = (3*(4*rho_I + 3*p_x1_x1_I*rho_I - 3*omegaVar*p_x1_x1_I*rho_I))/(omegaVar + 9);
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (2*rho_I + 15*p_x1_x1_I*rho_I)/(omegaVar + 9);
    p_x2_x2 = (40*p_x2_x2_I*rho_I)/33 - (4*p_x3_x3_I*rho_I)/33;
    p_x3_x3 = (40*p_x3_x3_I*rho_I)/33 - (4*p_x2_x2_I*rho_I)/33;
    p_x1_x2 = 2*p_x1_x2_I*rho_I;
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux2;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x2_x2;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x1_x3;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case EAST:
    l1 = 1;
    l2 = 0;
    l3 = 0;
    // u1 = ux // u2  = uy // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] + pop[12] - pop[17] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_xx_I;
    p_x2_x2_I = m_yy_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_xz_I;
    p_x2_x3_I = m_yz_I;

    rho = (3*(4*rho_I + 3*p_x1_x1_I*rho_I - 3*omegaVar*p_x1_x1_I*rho_I))/(omegaVar + 9);
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (2*rho_I + 15*p_x1_x1_I*rho_I)/(omegaVar + 9);
    p_x2_x2 = (40*p_x2_x2_I*rho_I)/33 - (4*p_x3_x3_I*rho_I)/33;
    p_x3_x3 = (40*p_x3_x3_I*rho_I)/33 - (4*p_x2_x2_I*rho_I)/33;
    p_x1_x2 = 2*p_x1_x2_I*rho_I;
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = p_x2_x3_I*rho_I;

    ux_t30 = ux1;
    uy_t30 = ux2;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x1_x1;
    m_yy_t45 = inv_rho * p_x2_x2;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x1_x3;
    m_yz_t90 = inv_rho * p_x2_x3;

    rhoVar = rho;

    break;
case SOUTH:
    l1 = -1;
    l2 = 0;
    l3 = 0;
    // u1 = uy // u2  = ux // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[8] - pop[13]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[12] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_yy_I;
    p_x2_x2_I = m_xx_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_yz_I;
    p_x2_x3_I = m_xz_I;

    rho = (3*(4*rho_I + 3*p_x1_x1_I*rho_I - 3*omegaVar*p_x1_x1_I*rho_I))/(omegaVar + 9);
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (2*rho_I + 15*p_x1_x1_I*rho_I)/(omegaVar + 9);
    p_x2_x2 = (40*p_x2_x2_I*rho_I)/33 - (4*p_x3_x3_I*rho_I)/33;
    p_x3_x3 = (40*p_x3_x3_I*rho_I)/33 - (4*p_x2_x2_I*rho_I)/33;
    p_x1_x2 = 2*p_x1_x2_I*rho_I;
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = p_x2_x3_I*rho_I;

    ux_t30 = ux2;
    uy_t30 = ux1;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x2_x2;
    m_yy_t45 = inv_rho * p_x1_x1;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x2_x3;
    m_yz_t90 = inv_rho * p_x1_x3;

    rhoVar = rho;

    break;
case NORTH:
    l1 = 1;
    l2 = 0;
    l3 = 0;
    // u1 = uy // u2  = ux // u3 == uz
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] + pop[10] - pop[15] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] - (1.0/3.0)*pop[5] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] - (1.0/3.0)*pop[9] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[11] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] + (2.0/3.0)*pop[5] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_yy_I;
    p_x2_x2_I = m_xx_I;
    p_x3_x3_I = m_zz_I;
    p_x1_x2_I = m_xy_I;
    p_x1_x3_I = m_yz_I;
    p_x2_x3_I = m_xz_I;

    rho = (3*(4*rho_I + 3*p_x1_x1_I*rho_I - 3*omegaVar*p_x1_x1_I*rho_I))/(omegaVar + 9);
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (2*rho_I + 15*p_x1_x1_I*rho_I)/(omegaVar + 9);
    p_x2_x2 = (40*p_x2_x2_I*rho_I)/33 - (4*p_x3_x3_I*rho_I)/33;
    p_x3_x3 = (40*p_x3_x3_I*rho_I)/33 - (4*p_x2_x2_I*rho_I)/33;
    p_x1_x2 = 2*p_x1_x2_I*rho_I;
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = p_x2_x3_I*rho_I;

    ux_t30 = ux2;
    uy_t30 = ux1;
    uz_t30 = ux3;

    m_xx_t45 = inv_rho * p_x2_x2;
    m_yy_t45 = inv_rho * p_x1_x1;
    m_zz_t45 = inv_rho * p_x3_x3;
    m_xy_t90 = inv_rho * p_x1_x2;
    m_xz_t90 = inv_rho * p_x2_x3;
    m_yz_t90 = inv_rho * p_x1_x3;

    rhoVar = rho;

    break;
case BACK:
    l1 = -1;
    l2 = 0;
    l3 = 0;
    // u1 = uz // u2  = ux // u3 == uy
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[10] - (1.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[15] - (1.0/3.0)*pop[17]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[10] - pop[15]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[6] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[10] + (2.0/3.0)*pop[12] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);
    m_yz_I = inv_rho_I * (pop[12] - pop[17]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[6] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[10] + (2.0/3.0)*pop[12] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[15] + (2.0/3.0)*pop[17]);

    p_x1_x1_I = m_zz_I;
    p_x2_x2_I = m_xx_I;
    p_x3_x3_I = m_yy_I;
    p_x1_x2_I = m_xz_I;
    p_x1_x3_I = m_yz_I;
    p_x2_x3_I = m_xy_I;

    rho = (3*(4*rho_I + 3*p_x1_x1_I*rho_I - 3*omegaVar*p_x1_x1_I*rho_I))/(omegaVar + 9);
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (2*rho_I + 15*p_x1_x1_I*rho_I)/(omegaVar + 9);
    p_x2_x2 = (40*p_x2_x2_I*rho_I)/33 - (4*p_x3_x3_I*rho_I)/33;
    p_x3_x3 = (40*p_x3_x3_I*rho_I)/33 - (4*p_x2_x2_I*rho_I)/33;
    p_x1_x2 = 2*p_x1_x2_I*rho_I;
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = p_x2_x3_I*rho_I;

    ux_t30 = ux2;
    uy_t30 = ux3;
    uz_t30 = ux1;

    m_xx_t45 = inv_rho * p_x2_x2;
    m_yy_t45 = inv_rho * p_x3_x3;
    m_zz_t45 = inv_rho * p_x1_x1;
    m_xy_t90 = inv_rho * p_x2_x3;
    m_xz_t90 = inv_rho * p_x1_x2;
    m_yz_t90 = inv_rho * p_x1_x3;

    rhoVar = rho;

    break;
case FRONT:
    l1 = 1;
    l2 = 0;
    l3 = 0;
    // u1 = uz // u2  = ux // u3 == uy
    ux1 = 0.0;
    ux2 = 0.0;
    ux3 = 0.0;

    rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    inv_rho_I = 1.0 / rho_I;
    m_xx_I = inv_rho_I * (-(1.0/3.0)*pop[0] + (2.0/3.0)*pop[1] + (2.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] + (2.0/3.0)*pop[9] - (1.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] + (2.0/3.0)*pop[16] - (1.0/3.0)*pop[18]);
    m_xy_I = inv_rho_I * (pop[7] + pop[8] - pop[13] - pop[14]);
    m_xz_I = inv_rho_I * (pop[9] - pop[16]);
    m_yy_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] + (2.0/3.0)*pop[3] + (2.0/3.0)*pop[4] - (1.0/3.0)*pop[5] + (2.0/3.0)*pop[7] + (2.0/3.0)*pop[8] - (1.0/3.0)*pop[9] + (2.0/3.0)*pop[11] + (2.0/3.0)*pop[13] + (2.0/3.0)*pop[14] - (1.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);
    m_yz_I = inv_rho_I * (pop[11] - pop[18]);
    m_zz_I = inv_rho_I * (-(1.0/3.0)*pop[0] - (1.0/3.0)*pop[1] - (1.0/3.0)*pop[2] - (1.0/3.0)*pop[3] - (1.0/3.0)*pop[4] + (2.0/3.0)*pop[5] - (1.0/3.0)*pop[7] - (1.0/3.0)*pop[8] + (2.0/3.0)*pop[9] + (2.0/3.0)*pop[11] - (1.0/3.0)*pop[13] - (1.0/3.0)*pop[14] + (2.0/3.0)*pop[16] + (2.0/3.0)*pop[18]);

    p_x1_x1_I = m_zz_I;
    p_x2_x2_I = m_xx_I;
    p_x3_x3_I = m_yy_I;
    p_x1_x2_I = m_xz_I;
    p_x1_x3_I = m_yz_I;
    p_x2_x3_I = m_xy_I;

    rho = (3*(4*rho_I + 3*p_x1_x1_I*rho_I - 3*omegaVar*p_x1_x1_I*rho_I))/(omegaVar + 9);
    inv_rho = 1.0 / rho;
    inv_rho2 = rho_I * inv_rho;

    p_x1_x1 = (2*rho_I + 15*p_x1_x1_I*rho_I)/(omegaVar + 9);
    p_x2_x2 = (40*p_x2_x2_I*rho_I)/33 - (4*p_x3_x3_I*rho_I)/33;
    p_x3_x3 = (40*p_x3_x3_I*rho_I)/33 - (4*p_x2_x2_I*rho_I)/33;
    p_x1_x2 = 2*p_x1_x2_I*rho_I;
    p_x1_x3 = 2*p_x1_x3_I*rho_I;
    p_x2_x3 = p_x2_x3_I*rho_I;

    ux_t30 = ux2;
    uy_t30 = ux3;
    uz_t30 = ux1;

    m_xx_t45 = inv_rho * p_x2_x2;
    m_yy_t45 = inv_rho * p_x3_x3;
    m_zz_t45 = inv_rho * p_x1_x1;
    m_xy_t90 = inv_rho * p_x2_x3;
    m_xz_t90 = inv_rho * p_x1_x2;
    m_yz_t90 = inv_rho * p_x1_x3;

    rhoVar = rho;

    break;

default:
        break;

rhoVar = rho;
}