rho = RHO_0 + (1.0/(16.0*cs2))*RHO_0*U_MAX*U_MAX*(cos((dfloat)2.0*(x) / L) + cos((dfloat)2.0*(y) / L))*(cos((dfloat)2.0*(z) / L) + 2.0);
ux =   U_MAX * sin((dfloat)(x) / L) * cos((dfloat)(y) / L) * cos((dfloat)(z) / L);
uy = - U_MAX * cos((dfloat)(x) / L) * sin((dfloat)(y) / L) * cos((dfloat)(z) / L);
uz = 0.0;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif