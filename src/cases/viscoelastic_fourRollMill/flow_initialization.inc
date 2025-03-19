rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

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


dfloat R = 16;
dfloat sigma = 3.0;    // Smoothing parameter
dfloat CX = (NX - 1) / 2; // Center X
dfloat CY = (NY - 1) / 2; // Center Y
dfloat CZ = (NZ - 1) / 2; // Center Y
dfloat dx = x - CX;
dfloat dy = y - CY;
dfloat dz = z - CY;
dfloat d2 = dx * dx + dy * dy;
dfloat d1 = dx * dx + dz * dz;
dfloat d3 = dy * dy + dz * dz;

dfloat range = 0.1;

/*
#ifdef A_XX_DIST
AxxVar = (CONF_ZERO-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_XY_DIST
AxyVar = (CONF_ZERO-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_XZ_DIST
AxzVar = (CONF_ZERO-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_YY_DIST
AyyVar = (CONF_ZERO-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_YZ_DIST
AyzVar = (CONF_ZERO-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_ZZ_DIST
AzzVar = (CONF_ZERO-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
*/