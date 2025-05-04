rho = RHO_0;
ux = 0;
uy = 0;
uz = 0;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif

dfloat ttemp = T_REFERENCE;//T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);

#ifdef SECOND_DIST 
    cVar = ttemp;
    qx_t30  = 0; 
    qy_t30  = 0; 
    qz_t30  = 0; 
#endif