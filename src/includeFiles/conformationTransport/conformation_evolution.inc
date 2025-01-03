
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
dfloat ADU_DUTA_xx_t30 = 2*AxxVar*duxdx_t30 + 2*AxyVar*duydx_t30 + 2*AxzVar*duzdx_t30;
dfloat ADU_DUTA_yy_t30 = 2*AxyVar*duxdy_t30 + 2*AyyVar*duydy_t30 + 2*AyzVar*duzdy_t30;
dfloat ADU_DUTA_zz_t30 = 2*AxzVar*duxdz_t30 + 2*AyzVar*duydz_t30 + 2*AzzVar*duzdz_t30;

dfloat ADU_DUTA_xy_t30 = AxxVar*duxdy_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + AxzVar*duzdy_t30 + AyyVar*duydx_t30 + AyzVar*duzdx_t30;
dfloat ADU_DUTA_xz_t30 = AxxVar*duxdz_t30 + AxzVar*duxdx_t30 + AxyVar*duydz_t30 + AxzVar*duzdz_t30 + AyzVar*duydx_t30 + AzzVar*duzdx_t30;
dfloat ADU_DUTA_yz_t30 = AxyVar*duxdz_t30 + AxzVar*duxdy_t30 + AyyVar*duydz_t30 + AyzVar*duydy_t30 + AyzVar*duzdz_t30 + AzzVar*duzdy_t30;

dfloat aa, bb;

#ifdef FENE_P
    //trace of A
    dfloat trA = AxxVar + AyyVar + AzzVar;
    aa = -inv_lambda/(1-trA/(fenep_re*fenep_re));
    bb = -inv_lambda/(1-3.0/(fenep_re*fenep_re));
    #ifdef OLDROYD_B
    aa = 1.0;
    bb = 1.0;
    #endif
#endif

//compute source term for the conformation transport
// G = (1/lambda)*(I-A) + (A . div U + div U^T . A) * 3.0
//TODO: maybe eq36 is wrong here and it should have  been: G = (1/lambda)*(b*I-a*A) + (A . div U + div U^T . A) * 3.0
dfloat G_xx = inv_lambda * (bb - aa*AxxVar) + ADU_DUTA_xx_t30/F_M_I_SCALE;
dfloat G_yy = inv_lambda * (bb - aa*AyyVar) + ADU_DUTA_yy_t30/F_M_I_SCALE;
dfloat G_zz = inv_lambda * (bb - aa*AzzVar) + ADU_DUTA_zz_t30/F_M_I_SCALE;

dfloat G_xy =  ADU_DUTA_xy_t30/F_M_I_SCALE - inv_lambda * aa * AxyVar;
dfloat G_xz =  ADU_DUTA_xz_t30/F_M_I_SCALE - inv_lambda * aa * AxzVar;
dfloat G_yz =  ADU_DUTA_yz_t30/F_M_I_SCALE - inv_lambda * aa * AyzVar;

fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_xx;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_xy;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_xz;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_yy;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_yz;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_zz;

//if(x == 60 && y == 60 && z == 60)
//    printf("step %d xx %f \n", step,G_xx);

//compute the soource term for mass-momentum transport, eq36 malaspinas
// rho.g = div * PI =  (mu/lambda)*(a*A - b*I)
//F = (mu/lambda) * div(A) - div(I); but div(I) = 0
//F = (mu/lambda)*div(A)

L_Fx += (dAxxdx + dAxydy + dAxzdz)*(nu_p*inv_lambda)*switch_conf; //zero just to debug for now
L_Fy += (dAxydx + dAyydy + dAyzdz)*(nu_p*inv_lambda)*switch_conf; //zero just to debug for now
L_Fz += (dAxzdx + dAyzdy + dAzzdz)*(nu_p*inv_lambda)*switch_conf; //zero just to debug for now


AxxVar += CONF_ZERO;
AxyVar += CONF_ZERO;
AxzVar += CONF_ZERO;
AyyVar += CONF_ZERO;
AyzVar += CONF_ZERO;
AzzVar += CONF_ZERO;



