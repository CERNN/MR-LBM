
/*
// Example of usage of random numbers for turbulence in parallel plates flow in z  
    dfloat y_visc = 6.59, ub_f = 15.6, uc_f = 18.2;
    // logaritimic velocity profile
    dfloat uz_log; 
    dfloat pos = (y < NY/2 ? y + 0.5 : NY - (y + 0.5));
    uz_log = -(uc_f*U_TAU)*(((pos-NY/2)/del)*((pos-NY/2)/del)) + (uc_f*U_TAU);
    
    uz = uz_log;
    ux = 0.0;
    uy = 0.0;
    rho = RHO_0;


    // perturbation
    dfloat pert = 0.1;
    int l = idxScalarGlobal(x, y, z);
    int Nt = NUMBER_LBM_NODES;
    uz += (ub_f*U_TAU)*pert*randomNumbers[l + z - Nt*((l + z) / Nt)];
    ux += (ub_f*U_TAU)*pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
    uy += (ub_f*U_TAU)*pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
*/ 


/*dfloat pert = 0.05;
int l = idxScalarGlobal(x, y, z);
int Nt = NUMBER_LBM_NODES;

ux += (U_MAX)*pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
uy += (U_MAX)*pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
uz += (U_MAX)*pert*randomNumbers[l + z - Nt*((l + z) / Nt)];*/

rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif
