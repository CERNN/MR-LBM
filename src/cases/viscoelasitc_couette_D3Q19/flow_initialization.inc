rho = RHO_0;
ux = ((dfloat)y)*2*U_MAX/((dfloat)(NY-1)) - U_MAX;
uy = U_0_Y;
uz = U_0_Z;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif


#ifdef A_XX_DIST
AxxVar = CONF_ZERO + 1.0 +  2*Weissenberg_number*Weissenberg_number;
Axx_qx_t30 = 3.0*AxxVar*(ux + 0.0);
Axx_qy_t30 = 3.0*AxxVar*(uy + 0.0);
Axx_qz_t30 = 3.0*AxxVar*(uz + 0.0);
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO + Weissenberg_number;
Axy_qx_t30 = 3.0*AxyVar*(ux + 0.0);
Axy_qy_t30 = 3.0*AxyVar*(uy + 0.0);
Axy_qz_t30 = 3.0*AxyVar*(uz + 0.0);
#endif
#ifdef A_XZ_DIST
AxzVar = CONF_ZERO;
Axz_qx_t30 = 3.0*AxzVar*(ux + 0.0);
Axz_qy_t30 = 3.0*AxzVar*(uy + 0.0);
Axz_qz_t30 = 3.0*AxzVar*(uz + 0.0);
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO + 1.0;
Ayy_qx_t30 = 3.0*AyyVar*(ux + 0.0);
Ayy_qy_t30 = 3.0*AyyVar*(uy + 0.0);
Ayy_qz_t30 = 3.0*AyyVar*(uz + 0.0);
#endif
#ifdef A_YZ_DIST
AyzVar = CONF_ZERO;
Ayz_qx_t30 = 3.0*AyzVar*(ux + 0.0);
Ayz_qy_t30 = 3.0*AyzVar*(uy + 0.0);
Ayz_qz_t30 = 3.0*AyzVar*(uz + 0.0);
#endif
#ifdef A_ZZ_DIST
AzzVar = CONF_ZERO + 1.0;
Azz_qx_t30 = 3.0*AzzVar*(ux + 0.0);
Azz_qy_t30 = 3.0*AzzVar*(uy + 0.0);
Azz_qz_t30 = 3.0*AzzVar*(uz + 0.0);
#endif

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif