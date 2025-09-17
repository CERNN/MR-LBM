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