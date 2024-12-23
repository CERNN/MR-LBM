rho = RHO_0;
ux = U_0_X;
uy = U_0_Y;
uz = U_0_Z;

dfloat xx = 2.0*M_PI*x/L;
dfloat yy = 2.0*M_PI*y/L;
dfloat zz = 2.0*M_PI*z/L;

//on malaspinas paper the theta is already accounted for.
dfloat thetaa = M_PI/2;

ux =  U_MAX*(2.0/sqrtf(3.0))*sin(theta + 2.0*M_PI/3.0)sin(xx)*cos(yy)*cos(zz);
uy =  U_MAX*(2.0/sqrtf(3.0))*sin(theta - 2.0*M_PI/3.0)cos(xx)*sin(yy)*cos(zz);
uz =  U_MAX*(2.0/sqrtf(3.0))*sin(theta)cos(xx)*cos(yy)*sin(zz);

//TODO however when deriving the poisson equation it does not include the theta term
// the RHO_0 + (U_MAX*U_MAX/cs2) comes from converting pressure to densisty. U_max comes from non-dimensional units
rho = RHO_0 + (U_MAX*U_MAX/cs2)*(1.0/6.0)*(0.25*cos(2*yy) + 0.25*cos(-2*yy+2*xx) + 0.25*cos(2*xx+2*yy)
                                         + cos(2*zz) + (1.0/24)*cos(-2*zz+2*yy) + (1.0/24)*cos( 2*zz+2*yy)
                                                     + (1.0/24)*cos( 2*zz+2*xx) + (1.0/24)*cos(-2*zz+2*xx) + 0.25*cos(2*xx) - 1);



#ifdef NON_NEWTONIAN_FLUID
omega = OMEGA;
#endif
