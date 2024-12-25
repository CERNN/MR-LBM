rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;


//ux = U_MAX;// * ((dfloat)y/(NY-1));
/*
dfloat R = 32;
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


#ifdef A_XX_DIST
AxxVar = (1.0-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_XY_DIST
AxyVar = (1.0-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_XZ_DIST
AxzVar = (1.0-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_YY_DIST
AyyVar = (1.0-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_YZ_DIST
AyzVar = (1.0-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef A_ZZ_DIST
AzzVar = (1.0-range/2.0) + range / (1.0 + exp((d3 - R * R) / (sigma * sigma)));
#endif
#ifdef THERMAL_MODEL
cVar = 0.1 + 1.0 / (1.0 + exp((d2 - R * R) / (sigma * sigma)));
#endif
*/


dfloat xx = 2.0*M_PI*x/L;
dfloat yy = 2.0*M_PI*y/L;
dfloat zz = 2.0*M_PI*z/L;

//on malaspinas paper the theta is already accounted for.
dfloat thetaa = M_PI/2;

ux =  U_MAX*(2.0/sqrtf(3.0))*sin(thetaa + 2.0*M_PI/3.0)*sin(xx)*cos(yy)*cos(zz);
uy =  U_MAX*(2.0/sqrtf(3.0))*sin(thetaa - 2.0*M_PI/3.0)*cos(xx)*sin(yy)*cos(zz);
uz =  U_MAX*(2.0/sqrtf(3.0))*sin(thetaa)*cos(xx)*cos(yy)*sin(zz);

//TODO however when deriving the poisson equation it does not include the theta term
// the RHO_0 + (U_MAX*U_MAX/cs2) comes from converting pressure to densisty. U_max comes from non-dimensional units
rho = RHO_0 + (U_MAX*U_MAX/cs2)*(1.0/6.0)*(0.25*cos(2*yy) + 0.25*cos(-2*yy+2*xx) + 0.25*cos(2*xx+2*yy)
                                         + cos(2*zz) + (1.0/24)*cos(-2*zz+2*yy) + (1.0/24)*cos( 2*zz+2*yy)
                                                     + (1.0/24)*cos( 2*zz+2*xx) + (1.0/24)*cos(-2*zz+2*xx) + 0.25*cos(2*xx) - 1);

#ifdef A_XX_DIST
AxxVar = CONF_ZERO;
#endif
#ifdef A_XY_DIST
AxyVar = CONF_ZERO;
#endif
#ifdef A_XZ_DIST
AxzVar = CONF_ZERO;
#endif
#ifdef A_YY_DIST
AyyVar = CONF_ZERO;
#endif
#ifdef A_YZ_DIST
AyzVar = CONF_ZERO;
#endif
#ifdef A_ZZ_DIST
AzzVar = CONF_ZERO;
#endif

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif
