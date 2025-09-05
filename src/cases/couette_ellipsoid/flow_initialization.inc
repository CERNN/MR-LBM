rho = RHO_0;
ux = ((dfloat)y)*2*U_MAX/((dfloat)(NY-1)) - U_MAX;
uy = U_0_Y;
uz = U_0_Z;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif
