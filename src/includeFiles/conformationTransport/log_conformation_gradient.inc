// Fetch neighboring values for Cxx, Cxy, Cxz, Cyy, Cyz, Czz

//store to be used in the next step
if(step%2 == 0){
    dfloat Cxx_xm1 = fMom[idxMom(txm1, ty, tz, C_XX_2_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
    dfloat Cxx_xp1 = fMom[idxMom(txp1, ty, tz, C_XX_2_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];
    
    dfloat Cxy_ym1 = fMom[idxMom(tx, tym1, tz, C_XY_2_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
    dfloat Cxy_yp1 = fMom[idxMom(tx, typ1, tz, C_XY_2_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];
    
    dfloat Cxz_zm1 = fMom[idxMom(tx, ty, tzm1, C_XZ_2_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
    dfloat Cxz_zp1 = fMom[idxMom(tx, ty, tzp1, C_XZ_2_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    
    dfloat Cxy_xm1 = fMom[idxMom(txm1, ty, tz, C_XY_2_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
    dfloat Cxy_xp1 = fMom[idxMom(txp1, ty, tz, C_XY_2_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];
    
    dfloat Cyy_ym1 = fMom[idxMom(tx, tym1, tz, C_YY_2_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
    dfloat Cyy_yp1 = fMom[idxMom(tx, typ1, tz, C_YY_2_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];
    
    dfloat Cyz_zm1 = fMom[idxMom(tx, ty, tzm1, C_YZ_2_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
    dfloat Cyz_zp1 = fMom[idxMom(tx, ty, tzp1, C_YZ_2_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    
    dfloat Cxz_xm1 = fMom[idxMom(txm1, ty, tz, C_XZ_2_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
    dfloat Cxz_xp1 = fMom[idxMom(txp1, ty, tz, C_XZ_2_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];
    
    dfloat Cyz_ym1 = fMom[idxMom(tx, tym1, tz, C_YZ_2_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
    dfloat Cyz_yp1 = fMom[idxMom(tx, typ1, tz, C_YZ_2_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];
    
    dfloat Czz_zm1 = fMom[idxMom(tx, ty, tzm1, C_ZZ_2_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
    dfloat Czz_zp1 = fMom[idxMom(tx, ty, tzp1, C_ZZ_2_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}else{
    dfloat Cxx_xm1 = fMom[idxMom(txm1, ty, tz, C_XX_1_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
    dfloat Cxx_xp1 = fMom[idxMom(txp1, ty, tz, C_XX_1_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];
    
    dfloat Cxy_ym1 = fMom[idxMom(tx, tym1, tz, C_XY_1_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
    dfloat Cxy_yp1 = fMom[idxMom(tx, typ1, tz, C_XY_1_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];
    
    dfloat Cxz_zm1 = fMom[idxMom(tx, ty, tzm1, C_XZ_1_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
    dfloat Cxz_zp1 = fMom[idxMom(tx, ty, tzp1, C_XZ_1_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    
    dfloat Cxy_xm1 = fMom[idxMom(txm1, ty, tz, C_XY_1_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
    dfloat Cxy_xp1 = fMom[idxMom(txp1, ty, tz, C_XY_1_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];
    
    dfloat Cyy_ym1 = fMom[idxMom(tx, tym1, tz, C_YY_1_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
    dfloat Cyy_yp1 = fMom[idxMom(tx, typ1, tz, C_YY_1_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];
    
    dfloat Cyz_zm1 = fMom[idxMom(tx, ty, tzm1, C_YZ_1_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
    dfloat Cyz_zp1 = fMom[idxMom(tx, ty, tzp1, C_YZ_1_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    
    dfloat Cxz_xm1 = fMom[idxMom(txm1, ty, tz, C_XZ_1_INDEX, ((tx == 0) ? bxm1 : bx) , by, bz)];
    dfloat Cxz_xp1 = fMom[idxMom(txp1, ty, tz, C_XZ_1_INDEX, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx), by, bz)];
    
    dfloat Cyz_ym1 = fMom[idxMom(tx, tym1, tz, C_YZ_1_INDEX, bx, ((ty == 0) ? bym1 : by), bz)];
    dfloat Cyz_yp1 = fMom[idxMom(tx, typ1, tz, C_YZ_1_INDEX, bx, ((ty == (BLOCK_NY - 1)) ? byp1 : by) , bz)];
    
    dfloat Czz_zm1 = fMom[idxMom(tx, ty, tzm1, C_ZZ_1_INDEX, bx, by, ((tz == 0) ? bzm1 : bz))];
    dfloat Czz_zp1 = fMom[idxMom(tx, ty, tzp1, C_ZZ_1_INDEX, bx, by, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}


// Compute derivatives
dfloat dCxxdx = (Cxx_xp1 - Cxx_xm1) / 2;
dfloat dCxydy = (Cxy_yp1 - Cxy_ym1) / 2;
dfloat dCxzdz = (Cxz_zp1 - Cxz_zm1) / 2;

dfloat dCxydx = (Cxy_xp1 - Cxy_xm1) / 2;
dfloat dCyydy = (Cyy_yp1 - Cyy_ym1) / 2;
dfloat dCyzdz = (Cyz_zp1 - Cyz_zm1) / 2;

dfloat dCxzdx = (Cxz_xp1 - Cxz_xm1) / 2;
dfloat dCyzdy = (Cyz_yp1 - Cyz_ym1) / 2;
dfloat dCzzdz = (Czz_zp1 - Czz_zm1) / 2;

if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dCxxdx = (Cxx_xp1 - CxxVar);
    dCxydx = (Cxy_xp1 - CxyVar);
    dCxzdx = (Cxz_xp1 - CxzVar);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dCxxdx = (CxxVar - Cxx_xm1);
    dCxydx = (CxyVar - Cxy_xm1);
    dCxzdx = (CxzVar - Cxz_xm1);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dCxydy = (Cxy_yp1 - CxyVar);
    dCyydy = (Cyy_yp1 - CyyVar);
    dCyzdy = (Cyz_yp1 - CyzVar);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dCxydy = (CxyVar - Cxy_ym1);
    dCyydy = (CyyVar - Cyy_ym1);
    dCyzdy = (CyzVar - Cyz_ym1);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dCxzdz = (Cxz_zp1 - CxzVar);
    dCyzdz = (Cyz_zp1 - CyzVar);
    dCzzdz = (Czz_zp1 - CzzVar);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dCxzdz = (CxzVar - Cxz_zm1);
    dCyzdz = (CyzVar - Cyz_zm1);
    dCzzdz = (CzzVar - Czz_zm1);
}
