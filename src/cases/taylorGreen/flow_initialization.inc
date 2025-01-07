rho = RHO_0 + (1.0/(16.0*cs2))*RHO_0*U_MAX*U_MAX*(cos(2*(x) / L) + cos(2*(y) / L))*(cos(2*(z) / L) + 2.0);
ux =   U_MAX * sin((x) / L) * cos((y) / L) * cos((z) / L);
uy = - U_MAX * cos((x) / L) * sin((y) / L) * cos((z) / L);
uz = 0.0;


#ifdef OMEGA_FIELD
omega = OMEGA;
#endif