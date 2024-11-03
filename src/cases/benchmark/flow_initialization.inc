rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

dfloat pert = 0.05;
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
uz += pert*randomNumbers[l + z - Nt*((l + z) / Nt)];
