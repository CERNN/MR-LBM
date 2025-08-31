// Fetch neighboring values for Axx, Axy, Axz, Ayy, Ayz, Azz
dfloat Axx_xm1 = fMom[idxMom(txm1, ty, tz, A_XX_C_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
dfloat Axx_xp1 = fMom[idxMom(txp1, ty, tz, A_XX_C_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];

dfloat Axy_ym1 = fMom[idxMom(tx, tym1, tz, A_XY_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Axy_yp1 = fMom[idxMom(tx, typ1, tz, A_XY_C_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];

dfloat Axz_zm1 = fMom[idxMom(tx, ty, tzm1, A_XZ_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Axz_zp1 = fMom[idxMom(tx, ty, tzp1, A_XZ_C_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];

dfloat Axy_xm1 = fMom[idxMom(txm1, ty, tz, A_XY_C_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
dfloat Axy_xp1 = fMom[idxMom(txp1, ty, tz, A_XY_C_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];

dfloat Ayy_ym1 = fMom[idxMom(tx, tym1, tz, A_YY_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Ayy_yp1 = fMom[idxMom(tx, typ1, tz, A_YY_C_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];

dfloat Ayz_zm1 = fMom[idxMom(tx, ty, tzm1, A_YZ_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Ayz_zp1 = fMom[idxMom(tx, ty, tzp1, A_YZ_C_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];

dfloat Axz_xm1 = fMom[idxMom(txm1, ty, tz, A_XZ_C_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
dfloat Axz_xp1 = fMom[idxMom(txp1, ty, tz, A_XZ_C_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];

dfloat Ayz_ym1 = fMom[idxMom(tx, tym1, tz, A_YZ_C_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
dfloat Ayz_yp1 = fMom[idxMom(tx, typ1, tz, A_YZ_C_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];

dfloat Azz_zm1 = fMom[idxMom(tx, ty, tzm1, A_ZZ_C_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
dfloat Azz_zp1 = fMom[idxMom(tx, ty, tzp1, A_ZZ_C_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];

// Compute derivatives
dfloat dAxxdx = (Axx_xp1 - Axx_xm1) *0.5;
dfloat dAxydy = (Axy_yp1 - Axy_ym1) *0.5;
dfloat dAxzdz = (Axz_zp1 - Axz_zm1) *0.5;

dfloat dAxydx = (Axy_xp1 - Axy_xm1) *0.5;
dfloat dAyydy = (Ayy_yp1 - Ayy_ym1) *0.5;
dfloat dAyzdz = (Ayz_zp1 - Ayz_zm1) *0.5;

dfloat dAxzdx = (Axz_xp1 - Axz_xm1) *0.5;
dfloat dAyzdy = (Ayz_yp1 - Ayz_ym1) *0.5;
dfloat dAzzdz = (Azz_zp1 - Azz_zm1) *0.5;

if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dAxxdx = (Axx_xp1 - AxxVar);
    dAxydx = (Axy_xp1 - AxyVar);
    dAxzdx = (Axz_xp1 - AxzVar);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dAxxdx = (AxxVar - Axx_xm1);
    dAxydx = (AxyVar - Axy_xm1);
    dAxzdx = (AxzVar - Axz_xm1);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dAxydy = (Axy_yp1 - AxyVar);
    dAyydy = (Ayy_yp1 - AyyVar);
    dAyzdy = (Ayz_yp1 - AyzVar);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dAxydy = (AxyVar - Axy_ym1);
    dAyydy = (AyyVar - Ayy_ym1);
    dAyzdy = (AyzVar - Ayz_ym1);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dAxzdz = (Axz_zp1 - AxzVar);
    dAyzdz = (Ayz_zp1 - AyzVar);
    dAzzdz = (Azz_zp1 - AzzVar);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dAxzdz = (AxzVar - Axz_zm1);
    dAyzdz = (AyzVar - Ayz_zm1);
    dAzzdz = (AzzVar - Azz_zm1);
}


/*s_pop[idxConfBlock(tx, ty, tz, 0)] = AxxVar;
s_pop[idxConfBlock(tx, ty, tz, 1)] = AxyVar;
s_pop[idxConfBlock(tx, ty, tz, 2)] = AxzVar;
s_pop[idxConfBlock(tx, ty, tz, 3)] = AyyVar;
s_pop[idxConfBlock(tx, ty, tz, 4)] = AyzVar;
s_pop[idxConfBlock(tx, ty, tz, 5)] = AzzVar;

__syncthreads();

dfloat Axx_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 0)];
dfloat Axy_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 1)];
dfloat Axz_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 2)];
dfloat Ayy_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 3)];
dfloat Ayz_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 4)];
dfloat Azz_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 5)];

dfloat Axx_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 0)];
dfloat Axy_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 1)];
dfloat Axz_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 2)];
dfloat Ayy_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 3)];
dfloat Ayz_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 4)];
dfloat Azz_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 5)];

dfloat Axx_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 0)];
dfloat Axy_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 1)];
dfloat Axz_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 2)];
dfloat Ayy_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 3)];
dfloat Ayz_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 4)];
dfloat Azz_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 5)];

dfloat Axx_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 0)];
dfloat Axy_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 1)];
dfloat Axz_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 2)];
dfloat Ayy_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 3)];
dfloat Ayz_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 4)];
dfloat Azz_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 5)];

dfloat Axx_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 0)];
dfloat Axy_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 1)];
dfloat Axz_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 2)];
dfloat Ayy_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 3)];
dfloat Ayz_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 4)];
dfloat Azz_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 5)];

dfloat Axx_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 0)];
dfloat Axy_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 1)];
dfloat Axz_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 2)];
dfloat Ayy_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 3)];
dfloat Ayz_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 4)];
dfloat Azz_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 5)];



// Compute unrolled gradient calculations
//FINITE CENTRAL DIFFERENCE ORDER 1, otherwise i would have to store a halo with width of 2 in shared memory

dfloat dAxxdx = (Axx_xp1 - Axx_xm1)/2;
dfloat dAxydx = (Axy_xp1 - Axy_xm1)/2;
dfloat dAxzdx = (Axz_xp1 - Axz_xm1)/2;
dfloat dAyydx = (Ayy_xp1 - Ayy_xm1)/2;
dfloat dAyzdx = (Ayz_xp1 - Ayz_xm1)/2;
dfloat dAzzdx = (Azz_xp1 - Azz_xm1)/2;

dfloat dAxxdy = (Axx_yp1 - Axx_ym1)/2;
dfloat dAxydy = (Axy_yp1 - Axy_ym1)/2;
dfloat dAxzdy = (Axz_yp1 - Axz_ym1)/2;
dfloat dAyydy = (Ayy_yp1 - Ayy_ym1)/2;
dfloat dAyzdy = (Ayz_yp1 - Ayz_ym1)/2;
dfloat dAzzdy = (Azz_yp1 - Azz_ym1)/2;

dfloat dAxxdz = (Axx_zp1 - Axx_zm1)/2;
dfloat dAxydz = (Axy_zp1 - Axy_zm1)/2;
dfloat dAxzdz = (Axz_zp1 - Axz_zm1)/2;
dfloat dAyydz = (Ayy_zp1 - Ayy_zm1)/2;
dfloat dAyzdz = (Ayz_zp1 - Ayz_zm1)/2;
dfloat dAzzdz = (Azz_zp1 - Azz_zm1)/2;

if (tx == 0) {   
    dAxxdx = (Axx_xp1 - ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 0, bxm1, by, bz)]) / 2;                        
    dAxydx = (Axy_xp1 - ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 1, bxm1, by, bz)]) / 2;                        
    dAxzdx = (Axz_xp1 - ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 2, bxm1, by, bz)]) / 2;                        
    dAyydx = (Ayy_xp1 - ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 3, bxm1, by, bz)]) / 2;                        
    dAyzdx = (Ayz_xp1 - ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 4, bxm1, by, bz)]) / 2;                        
    dAzzdx = (Azz_xp1 - ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 5, bxm1, by, bz)]) / 2;                        
}
if (tx == BLOCK_NX - 1) {
    dAxxdx = (ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 0, bxp1, by, bz)] - Axx_xm1) / 2;
    dAxydx = (ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 1, bxp1, by, bz)] - Axy_xm1) / 2;
    dAxzdx = (ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 2, bxp1, by, bz)] - Axz_xm1) / 2;
    dAyydx = (ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 3, bxp1, by, bz)] - Ayy_xm1) / 2;
    dAyzdx = (ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 4, bxp1, by, bz)] - Ayz_xm1) / 2;
    dAzzdx = (ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 5, bxp1, by, bz)] - Azz_xm1) / 2;
}

if (ty == 0) {   
    dAxxdy = (Axx_yp1 - ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 0, bx, bym1, bz)]) / 2;
    dAxydy = (Axy_yp1 - ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 1, bx, bym1, bz)]) / 2;
    dAxzdy = (Axz_yp1 - ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 2, bx, bym1, bz)]) / 2;                              
    dAyydy = (Ayy_yp1 - ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 3, bx, bym1, bz)]) / 2;                              
    dAyzdy = (Ayz_yp1 - ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 4, bx, bym1, bz)]) / 2;                              
    dAzzdy = (Azz_yp1 - ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 5, bx, bym1, bz)]) / 2;                              
}
if (ty == BLOCK_NY - 1) {
    dAxxdy = (ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 0, bx, byp1, bz)] - Axx_ym1) / 2;
    dAxydy = (ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 1, bx, byp1, bz)] - Axy_ym1) / 2;
    dAxzdy = (ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 2, bx, byp1, bz)] - Axz_ym1) / 2;
    dAyydy = (ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 3, bx, byp1, bz)] - Ayy_ym1) / 2;
    dAyzdy = (ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 4, bx, byp1, bz)] - Ayz_ym1) / 2;
    dAzzdy = (ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 5, bx, byp1, bz)] - Azz_ym1) / 2;
}

if (tz == 0) {   
    dAxxdz = (Axx_zp1 - ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 0, bx, by, bzm1)]) / 2;
    dAxydz = (Axy_zp1 - ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 1, bx, by, bzm1)]) / 2;
    dAxzdz = (Axz_zp1 - ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 2, bx, by, bzm1)]) / 2;                              
    dAyydz = (Ayy_zp1 - ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 3, bx, by, bzm1)]) / 2;                              
    dAyzdz = (Ayz_zp1 - ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 4, bx, by, bzm1)]) / 2;                              
    dAzzdz = (Azz_zp1 - ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 5, bx, by, bzm1)]) / 2;                              
}
if (tz == BLOCK_NZ - 1) {
    dAxxdz = (ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 0, bx, by, bzp1)] - Axx_zm1) / 2;
    dAxydz = (ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 1, bx, by, bzp1)] - Axy_zm1) / 2;
    dAxzdz = (ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 2, bx, by, bzp1)] - Axz_zm1) / 2;
    dAyydz = (ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 3, bx, by, bzp1)] - Ayy_zm1) / 2;
    dAyzdz = (ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 4, bx, by, bzp1)] - Ayz_zm1) / 2;
    dAzzdz = (ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 5, bx, by, bzp1)] - Azz_zm1) / 2;
}



*/
//if(x == 32 && z == 32)
//    printf("y: %d dAyydy %f Ayy_p1 %f Ayy %f Ayy_m1 %f  \n",y,dAyydy,s_pop[idxConfBlock(tx, ty + 1, tz, 3)],s_pop[idxConfBlock(tx, ty, tz, 3)],s_pop[idxConfBlock(tx, ty - 1, tz, 3)]);
//
//OVERWRITE FOR NEAR WALL NODES
/*
if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dAxxdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 0)] - s_pop[idxConfBlock(tx, ty, tz, 0)]);
    dAxydx = (s_pop[idxConfBlock(tx + 1, ty, tz, 1)] - s_pop[idxConfBlock(tx, ty, tz, 1)]);
    dAxzdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 2)] - s_pop[idxConfBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dAxxdx = (s_pop[idxConfBlock(tx, ty, tz, 0)] - s_pop[idxConfBlock(tx - 1, ty, tz, 0)]);
    dAxydx = (s_pop[idxConfBlock(tx, ty, tz, 1)] - s_pop[idxConfBlock(tx - 1, ty, tz, 1)]);
    dAxzdx = (s_pop[idxConfBlock(tx, ty, tz, 2)] - s_pop[idxConfBlock(tx - 1, ty, tz, 2)]);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dAxydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 1)] - s_pop[idxConfBlock(tx, ty, tz, 1)]);
    dAyydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 3)] - s_pop[idxConfBlock(tx, ty, tz, 3)]);
    dAyzdy = (s_pop[idxConfBlock(tx, ty + 1, tz, 4)] - s_pop[idxConfBlock(tx, ty, tz, 4)]);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dAxydy = (s_pop[idxConfBlock(tx, ty, tz, 1)] - s_pop[idxConfBlock(tx, ty - 1, tz, 1)]);
    dAyydy = (s_pop[idxConfBlock(tx, ty, tz, 3)] - s_pop[idxConfBlock(tx, ty - 1, tz, 3)]);
    dAyzdy = (s_pop[idxConfBlock(tx, ty, tz, 4)] - s_pop[idxConfBlock(tx, ty - 1, tz, 4)]);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dAxzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 2)] - s_pop[idxConfBlock(tx, ty, tz, 2)]);
    dAyzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 4)] - s_pop[idxConfBlock(tx, ty, tz, 4)]);
    dAzzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 5)] - s_pop[idxConfBlock(tx, ty, tz, 5)]);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dAxzdz = (s_pop[idxConfBlock(tx, ty, tz, 2)] - s_pop[idxConfBlock(tx, ty, tz - 1, 2)]);
    dAyzdz = (s_pop[idxConfBlock(tx, ty, tz, 4)] - s_pop[idxConfBlock(tx, ty, tz - 1, 4)]);
    dAzzdz = (s_pop[idxConfBlock(tx, ty, tz, 5)] - s_pop[idxConfBlock(tx, ty, tz - 1, 5)]);
}
*/

//if(x == 4 && y == 4 && z == 4 )
//    printf("step %d dAxxdx %f \n",step,dAxxdx);