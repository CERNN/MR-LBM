rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

dfloat pert = 0.02;
int l = idxScalarGlobal(x, y, z);
int Nt = NUMBER_LBM_NODES;
dfloat xc = (NX-1)/2.0;
dfloat yc = (NY-1)/2.0;

dfloat distX = (x-xc)/(NX-1);
dfloat distY = (y-yc)/(NY-1);

dfloat velX = distY * pert;
dfloat velY = distY * pert;


ux += pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
uy += pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
uz += 0*randomNumbers[l + z - Nt*((l + z) / Nt)];

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif

//dfloat x0 = (NX - 1) / 2.0;
//dfloat k = 20.0 / (NX - 1);
//dfloat ttemp = T_HOT - T_DELTA_T / (1.0 + std::exp(-k * (x - x0)));
dfloat ttemp = T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
#ifdef SECOND_DIST 
    cVar = ttemp;
    qx_t30  = 0; 
    qy_t30  = 0; 
    qz_t30  = 0; 
#endif