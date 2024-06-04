dfloat ux = ux_t30 + L_Fx*invRho/2;
dfloat uy = uy_t30 + L_Fy*invRho/2;
dfloat uz = uz_t30 + L_Fz*invRho/2;


dfloat m_xy = T_OMEGA * m_xy_t90 + OMEGA*ux_t30*uy_t30 + TT_OMEGA * invRho * (L_Fx * uy_t30 + L_Fy * ux_t30);
dfloat m_xz = T_OMEGA * m_xz_t90 + OMEGA*ux_t30*uz_t30 + TT_OMEGA * invRho * (L_Fx * uz_t30 + L_Fz * ux_t30);
dfloat m_yz = T_OMEGA * m_yz_t90 + OMEGA*uy_t30*uz_t30 + TT_OMEGA * invRho * (L_Fy * uz_t30 + L_Fz * uy_t30);

dfloat m_xx = ONETHIRD* (T_OMEGA * (2*m_xx_t45 - m_yy_t45 - m_zz_t45) +  (ux_t30*ux_t30 + uy_t30*uy_t30 + uz_t30*uz_t30) + OMEGA*(2*ux_t30*ux_t30 - uy_t30*uy_t30 - uz_t30*uz_t30) + invRho*T_OMEGA*(L_Fx*ux_t30*2 - L_Fy*uy_t30 - L_Fz*uz_t30)) + invRho*L_Fx*ux_t30;
dfloat m_yy = ONETHIRD* (T_OMEGA * (2*m_yy_t45 - m_xx_t45 - m_zz_t45) +  (ux_t30*ux_t30 + uy_t30*uy_t30 + uz_t30*uz_t30) + OMEGA*(2*uy_t30*uy_t30 - ux_t30*ux_t30 - uz_t30*uz_t30) + invRho*T_OMEGA*(L_Fy*uy_t30*2 - L_Fx*ux_t30 - L_Fz*uz_t30)) + invRho*L_Fy*uy_t30;
dfloat m_zz = ONETHIRD* (T_OMEGA * (2*m_zz_t45 - m_xx_t45 - m_yy_t45) +  (ux_t30*ux_t30 + uy_t30*uy_t30 + uz_t30*uz_t30) + OMEGA*(2*uz_t30*uz_t30 - ux_t30*ux_t30 - uy_t30*uy_t30) + invRho*T_OMEGA*(L_Fz*uz_t30*2 - L_Fx*ux_t30 - L_Fy*uy_t30)) + invRho*L_Fz*uz_t30;