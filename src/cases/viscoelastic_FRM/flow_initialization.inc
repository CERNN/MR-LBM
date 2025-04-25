rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif


#ifdef A_XX_DIST
AxxVar = CONF_ZERO + 1.0;
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO;
#endif
#ifdef A_XZ_DIST
AxzVar = CONF_ZERO;
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO + 1.0;
#endif
#ifdef A_YZ_DIST
AyzVar = CONF_ZERO;
#endif
#ifdef A_ZZ_DIST
AzzVar = CONF_ZERO + 1.0;
#endif

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif