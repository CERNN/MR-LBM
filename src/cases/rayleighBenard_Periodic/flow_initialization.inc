rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

dfloat pert = 0.001;
int l = idxScalarGlobal(x, y, z);
int Nt = NUMBER_LBM_NODES;
ux += 0*randomNumbers[l + x - Nt*((l + x) / Nt)];
uy += 0*randomNumbers[l + y - Nt*((l + y) / Nt)];
uz += 0*randomNumbers[l + z - Nt*((l + z) / Nt)];

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif

dfloat ttemp = T_REFERENCE;//T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);

#ifdef SECOND_DIST 
    cVar = ttemp;
    qx_t30  = 0; 
    qy_t30  = 0; 
    qz_t30  = 0; 
#endif