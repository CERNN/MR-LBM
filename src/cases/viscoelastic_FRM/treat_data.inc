checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());


int xx = (NX-1)/2;
int yy = (NY-1)/2;
int zz = (NZ-1)/2;

int x_m1 = xx - 1;
int x_p1 = xx + 1;

int y_m1 = yy - 1;
int y_p1 = yy + 1;


dfloat ux_t30 = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];
dfloat uy_t30 = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];

dfloat ux_xm1 = h_fMom[idxMom(x_m1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, x_m1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];
dfloat ux_xp1 = h_fMom[idxMom(x_p1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, x_p1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];

dfloat ux_ym1 = h_fMom[idxMom(xx%BLOCK_NX, y_m1%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, y_m1/BLOCK_NY, zz/BLOCK_NZ)];
dfloat ux_yp1 = h_fMom[idxMom(xx%BLOCK_NX, y_p1%BLOCK_NY, zz%BLOCK_NZ, M_UX_INDEX, xx/BLOCK_NX, y_p1/BLOCK_NY, zz/BLOCK_NZ)];

dfloat uy_xm1 = h_fMom[idxMom(x_m1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, x_m1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];
dfloat uy_xp1 = h_fMom[idxMom(x_p1%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, x_p1/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)];

dfloat uy_ym1 = h_fMom[idxMom(xx%BLOCK_NX, y_m1%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, y_m1/BLOCK_NY, zz/BLOCK_NZ)];
dfloat uy_yp1 = h_fMom[idxMom(xx%BLOCK_NX, y_p1%BLOCK_NY, zz%BLOCK_NZ, M_UY_INDEX, xx/BLOCK_NX, y_p1/BLOCK_NY, zz/BLOCK_NZ)];

/*
dfloat duxdx_t30 = (ux_xp1 - ux_xm1) / 2;
dfloat duydx_t30 = (uy_xp1 - uy_xm1) / 2;

dfloat duxdy_t30 = (ux_yp1 - ux_ym1) / 2;
dfloat duydy_t30 = (uy_yp1 - uy_ym1) / 2;
*/
/* second order
dfloat duxdx_t30 = (ux_xp1 + ux_xm1 - 2*ux_t30);
dfloat duydx_t30 = (uy_xp1 + uy_xm1 - 2*uy_t30);

dfloat duxdy_t30 = (ux_yp1 + ux_ym1 + 2*ux_t30);
dfloat duydy_t30 = (uy_yp1 + uy_ym1 + 2*uy_t30);
*/

// When doing the central point at (N-1)/2  if the even number will cause the central do move down
// ie. N = 8 , center at 3.5, but doesnt exist a lattice, so derivative is between 3 and  4
dfloat duxdx_t30 = (ux_xp1 - ux_t30);
dfloat duydx_t30 = (uy_xp1 - uy_t30);

dfloat duxdy_t30 = (ux_yp1 - ux_t30);
dfloat duydy_t30 = (uy_yp1 - uy_t30);

// 3 because the stored field is 3 timex
// U_max because need to normality velcoty
// N/(2*M_PI) because the domain is from 0 to 2*M_PI
const dfloat epsi_c = 3.0*U_MAX/(N/(2*M_PI));

dfloat Axx = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, A_XX_C_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)] - CONF_ZERO;
dfloat Axy = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, A_XY_C_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)] - CONF_ZERO;
dfloat Ayy = h_fMom[idxMom(xx%BLOCK_NX, yy%BLOCK_NY, zz%BLOCK_NZ, A_YY_C_INDEX, xx/BLOCK_NX, yy/BLOCK_NY, zz/BLOCK_NZ)] - CONF_ZERO;

std::ostringstream strDataInfo("");
strDataInfo << std::scientific;
strDataInfo << std::setprecision(6);
                        //step           // epsilon                         /Wi_eff                                // Axx     //Axy          //Ayy;
strDataInfo <<"step,"<< step/t_c << "," << duxdx_t30/epsi_c << "," << Weissenberg_number*duxdx_t30/epsi_c << "," << Axx << "," << Axy  << "," << Ayy;
saveTreatData("_data_000",strDataInfo.str(),step);