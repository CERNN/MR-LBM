        dfloat ux = ux_t30 + L_Fx*invRho/2;
        dfloat uy = uy_t30 + L_Fy*invRho/2;
        dfloat uz = uz_t30 + L_Fz*invRho/2;

        dfloat m_xy = T_OMEGA * m_xy_t90 + OMEGA*ux_t30*uy_t30 + TT_OMEGA * invRho * (L_Fx * uy_t30 + L_Fy * ux_t30);
        dfloat m_xz = T_OMEGA * m_xz_t90 + OMEGA*ux_t30*uz_t30 + TT_OMEGA * invRho * (L_Fx * uz_t30 + L_Fz * ux_t30);
        dfloat m_yz = T_OMEGA * m_yz_t90 + OMEGA*uy_t30*uz_t30 + TT_OMEGA * invRho * (L_Fy * uz_t30 + L_Fz * uy_t30);

        dfloat m_xx 
        dfloat m_yy
        dfloat m_zz