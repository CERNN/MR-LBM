
rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

#ifdef OMEGA_FIELD
omega = OMEGA;
#endif

if (nodeType == 0b11111111){
    rho = RHO_0;
    ux = 0;
    uy = 0;
    uz = 0;
}
