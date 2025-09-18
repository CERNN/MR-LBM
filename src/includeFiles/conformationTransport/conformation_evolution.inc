
//remove offset
AxxVar -= CONF_ZERO;
AxyVar -= CONF_ZERO;
AxzVar -= CONF_ZERO;
AyyVar -= CONF_ZERO;
AyzVar -= CONF_ZERO;
AzzVar -= CONF_ZERO;

dfloat switch_conf = 0.0;
if (step>CONFORMATION_SWITCH){
    switch_conf = 1.0;     
}

// (A . div U + div U^T . A) * 3.0
/*
dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_t30 + 2*AxyVar*duxdy_t30 + 2*0*0;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_t30 + 2*AyyVar*duydy_t30 + 2*0*0;
dfloat ADU_DUTA_zz_t30 = 2*0*0 + 2*0*0 + 2*0*0;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + 0*0 + AyyVar*duxdy_t30 + 0*0;
dfloat ADU_DUTA_xz_t30 = AxxVar*0 + 0*duxdx_t30 + AxyVar*0 + 0*0 + 0*duxdy_t30 + 0*0;
dfloat ADU_DUTA_yz_t30 = AxyVar*0 + 0*duydx_t30 + AyyVar*0 + 0*duydy_t30 + 0*0 + 0*0;
*/

dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_t30 + 2*AxyVar*duxdy_t30 + 2*AxzVar*duxdz_t30;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duydx_t30 + 2*AyyVar*duydy_t30 + 2*AyzVar*duydz_t30;
dfloat ADU_DUTA_zz_t30 = 2*AxzVar*duzdx_t30 + 2*AyzVar*duzdy_t30 + 2*AzzVar*duzdz_t30;

dfloat ADU_DUTA_xy_t30 = AxxVar*duydx_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + AxzVar*duydz_t30 + AyyVar*duxdy_t30 + AyzVar*duxdz_t30;
dfloat ADU_DUTA_xz_t30 = AxxVar*duzdx_t30 + AxzVar*duxdx_t30 + AxyVar*duzdy_t30 + AxzVar*duzdz_t30 + AyzVar*duxdy_t30 + AzzVar*duxdz_t30;
dfloat ADU_DUTA_yz_t30 = AxyVar*duzdx_t30 + AxzVar*duydx_t30 + AyyVar*duzdy_t30 + AyzVar*duydy_t30 + AyzVar*duzdz_t30 + AzzVar*duydz_t30;

dfloat aa, bb;

#ifdef FENE_P
    //trace of A
    dfloat trA = AxxVar + AyyVar + 0;
    aa = -inv_lambda/(1-trA/(fenep_re*fenep_re));
    bb = -inv_lambda/(1-3.0/(fenep_re*fenep_re));
    #ifdef OLDROYD_B
    aa = 1.0;
    bb = 1.0;
    #endif
#endif

aa = 1.0;
bb = 1.0;

//compute source term for the conformation transport
// G = (1/lambda)*(I-A) + (A . div U + div U^T . A) * 3.0
//TODO: maybe eq36 is wrong here and it should have  been: G = (1/lambda)*(b*I-a*A) + (A . div U + div U^T . A) * 3.0
dfloat Gxx = inv_lambda * (bb - aa*AxxVar) + ADU_DUTA_xx_t30/F_M_I_SCALE;
dfloat Gyy = inv_lambda * (bb - aa*AyyVar) + ADU_DUTA_yy_t30/F_M_I_SCALE;
dfloat Gzz = inv_lambda * (bb - aa*AzzVar) + ADU_DUTA_zz_t30/F_M_I_SCALE;

dfloat Gxy =  ADU_DUTA_xy_t30/F_M_I_SCALE - inv_lambda * aa * AxyVar;
dfloat Gxz =  ADU_DUTA_xz_t30/F_M_I_SCALE - inv_lambda * aa * AxzVar;
dfloat Gyz =  ADU_DUTA_yz_t30/F_M_I_SCALE - inv_lambda * aa * AyzVar;

//compute the soource term for mass-momentum transport, eq36 malaspinas
// rho.g = div * PI =  (mu/lambda)*(a*A - b*I)
//F = (mu/lambda) * div(A) - div(I); but div(I) = 0
//F = (mu/lambda)*div(A)

L_Fx += (dAxxdx + dAxydy + dAxzdz)*(nu_p*inv_lambda)*switch_conf;
L_Fy += (dAxydx + dAyydy + dAyzdz)*(nu_p*inv_lambda)*switch_conf;
L_Fz += (dAxzdx + dAyzdy + dAzzdz)*(nu_p*inv_lambda)*switch_conf;


AxxVar += CONF_ZERO;
AxyVar += CONF_ZERO;
AxzVar += CONF_ZERO;
AyyVar += CONF_ZERO;
AyzVar += CONF_ZERO;
AzzVar += CONF_ZERO;



