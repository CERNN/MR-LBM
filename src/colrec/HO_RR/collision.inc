dfloat ux = ux_t30 + L_Fx*invRho/2;
dfloat uy = uy_t30 + L_Fy*invRho/2;
dfloat uz = uz_t30 + L_Fz*invRho/2;

//matlab original
dfloat m_xx = (ux_t30*ux_t30 - (9*uy_t30*uy_t30*uz_t30*uz_t30)/2 + (3*m_zz_t45*uy_t30*uy_t30)/4 + 3*m_yz_t90*uy_t30*uz_t30 + (3*m_yy_t45*uz_t30*uz_t30)/4 - m_xx_t45)*omegaVar + ((15*uy_t30*uy_t30*uz_t30*uz_t30)/4 - (3*m_zz_t45*uy_t30*uy_t30)/4 - 3*m_yz_t90*uy_t30*uz_t30 - (3*m_yy_t45*uz_t30*uz_t30)/4 + m_xx_t45);
dfloat m_yy = ((3*m_zz_t45*ux_t30*ux_t30)/4 - (9*ux_t30*ux_t30*uz_t30*uz_t30)/2 + 3*m_xz_t90*ux_t30*uz_t30 + uy_t30*uy_t30 + (3*m_xx_t45*uz_t30*uz_t30)/4 - m_yy_t45)*omegaVar + ((15*ux_t30*ux_t30*uz_t30*uz_t30)/4 - (3*m_zz_t45*ux_t30*ux_t30)/4 - 3*m_xz_t90*ux_t30*uz_t30 - (3*m_xx_t45*uz_t30*uz_t30)/4 + m_yy_t45);
dfloat m_zz = ((3*m_yy_t45*ux_t30*ux_t30)/4 - (9*ux_t30*ux_t30*uy_t30*uy_t30)/2 + 3*m_xy_t90*ux_t30*uy_t30 + (3*m_xx_t45*uy_t30*uy_t30)/4 + uz_t30*uz_t30 - m_zz_t45)*omegaVar + ((15*ux_t30*ux_t30*uy_t30*uy_t30)/4 - (3*m_yy_t45*ux_t30*ux_t30)/4 - 3*m_xy_t90*ux_t30*uy_t30 - (3*m_xx_t45*uy_t30*uy_t30)/4 + m_zz_t45);
dfloat m_xy = (ux_t30*uy_t30 - m_xy_t90)*omegaVar + m_xy_t90;
dfloat m_xz = (ux_t30*uz_t30 - m_xz_t90)*omegaVar + m_xz_t90;
dfloat m_yz = (uy_t30*uz_t30 - m_yz_t90)*omegaVar + m_yz_t90;


//dfloat m_xx = ((((m_zz_t45*uy_t30*uy_t30 + m_yy_t45*uz_t30*uz_t30) - 6*uy_t30*uy_t30*uz_t30*uz_t30)/4 + m_yz_t90*uy_t30*uz_t30)*3 + ux_t30*ux_t30 - m_xx_t45)*omegaVar + (((5*uy_t30*uy_t30*uz_t30*uz_t30 - m_zz_t45*uy_t30*uy_t30 - m_yy_t45*uz_t30*uz_t30)/4 - m_yz_t90*uy_t30*uz_t30)*3 + m_xx_t45);
//dfloat m_yy = ((((m_zz_t45*ux_t30*ux_t30 + m_xx_t45*uz_t30*uz_t30) - 6*ux_t30*ux_t30*uz_t30*uz_t30)/4 + m_xz_t90*ux_t30*uz_t30)*3 + uy_t30*uy_t30 - m_yy_t45)*omegaVar + (((5*ux_t30*ux_t30*uz_t30*uz_t30 - m_zz_t45*ux_t30*ux_t30 - m_xx_t45*uz_t30*uz_t30)/4 - m_xz_t90*ux_t30*uz_t30)*3 + m_yy_t45);
//dfloat m_zz = ((((m_yy_t45*ux_t30*ux_t30 + m_xx_t45*uy_t30*uy_t30) - 6*ux_t30*ux_t30*uy_t30*uy_t30)/4 + m_xy_t90*ux_t30*uy_t30)*3 + uz_t30*uz_t30 - m_zz_t45)*omegaVar + (((5*ux_t30*ux_t30*uy_t30*uy_t30 - m_yy_t45*ux_t30*ux_t30 - m_xx_t45*uy_t30*uy_t30)/4 - m_xy_t90*ux_t30*uy_t30)*3 + m_zz_t45);
//dfloat m_xy = (ux_t30*uy_t30 - m_xy_t90)*omegaVar + m_xy_t90;
//dfloat m_xz = (ux_t30*uz_t30 - m_xz_t90)*omegaVar + m_xz_t90;
//dfloat m_yz = (uy_t30*uz_t30 - m_yz_t90)*omegaVar + m_yz_t90;