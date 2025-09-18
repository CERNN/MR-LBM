//THIS WORKS but is not efficient

dfloat ux_xm1 = fMom[idxMom(txm1, ty, tz, M_UX_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];
dfloat uy_xm1 = fMom[idxMom(txm1, ty, tz, M_UY_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];
dfloat uz_xm1 = fMom[idxMom(txm1, ty, tz, M_UZ_INDEX, ((x - 1 + NX) % NX) / BLOCK_NX, by, bz)];

dfloat ux_xp1 = fMom[idxMom(txp1, ty, tz, M_UX_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];
dfloat uy_xp1 = fMom[idxMom(txp1, ty, tz, M_UY_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];
dfloat uz_xp1 = fMom[idxMom(txp1, ty, tz, M_UZ_INDEX, ((x + 1) % NX) / BLOCK_NX, by, bz)];

dfloat ux_ym1 = fMom[idxMom(tx, tym1, tz, M_UX_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];
dfloat uy_ym1 = fMom[idxMom(tx, tym1, tz, M_UY_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];
dfloat uz_ym1 = fMom[idxMom(tx, tym1, tz, M_UZ_INDEX, bx, ((y - 1 + NY) % NY) / BLOCK_NY, bz)];

dfloat ux_yp1 = fMom[idxMom(tx, typ1, tz, M_UX_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];
dfloat uy_yp1 = fMom[idxMom(tx, typ1, tz, M_UY_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];
dfloat uz_yp1 = fMom[idxMom(tx, typ1, tz, M_UZ_INDEX, bx, ((y + 1) % NY) / BLOCK_NY, bz)];

dfloat ux_zm1 = fMom[idxMom(tx, ty, tzm1, M_UX_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];
dfloat uy_zm1 = fMom[idxMom(tx, ty, tzm1, M_UY_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];
dfloat uz_zm1 = fMom[idxMom(tx, ty, tzm1, M_UZ_INDEX, bx, by, ((z - 1 + NZ) % NZ) / BLOCK_NZ)];

dfloat ux_zp1 = fMom[idxMom(tx, ty, tzp1, M_UX_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];
dfloat uy_zp1 = fMom[idxMom(tx, ty, tzp1, M_UY_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];
dfloat uz_zp1 = fMom[idxMom(tx, ty, tzp1, M_UZ_INDEX, bx, by, ((z + 1) % NZ) / BLOCK_NZ)];


dfloat duxdx_t30 = (ux_xp1 - ux_xm1) *0.5;
dfloat duydx_t30 = (uy_xp1 - uy_xm1) *0.5;
dfloat duzdx_t30 = (uz_xp1 - uz_xm1) *0.5;

dfloat duxdy_t30 = (ux_yp1 - ux_ym1) *0.5;
dfloat duydy_t30 = (uy_yp1 - uy_ym1) *0.5;
dfloat duzdy_t30 = (uz_yp1 - uz_ym1) *0.5;

dfloat duxdz_t30 = (ux_zp1 - ux_zm1) *0.5;
dfloat duydz_t30 = (uy_zp1 - uy_zm1) *0.5;
dfloat duzdz_t30 = (uz_zp1 - uz_zm1) *0.5;

if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    duxdx_t30 = (ux_xp1 - ux_t30);
    duydx_t30 = (uy_xp1 - uy_t30);
    duzdx_t30 = (uz_xp1 - uz_t30);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    duxdx_t30 = (ux_t30 - ux_xm1);
    duydx_t30 = (uy_t30 - uy_xm1);
    duzdx_t30 = (uz_t30 - uz_xm1);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    duxdy_t30 = (ux_yp1 - ux_t30);
    duydy_t30 = (uy_yp1 - uy_t30);
    duzdy_t30 = (uz_yp1 - uz_t30);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    duxdy_t30 = (ux_t30 - ux_ym1);
    duydy_t30 = (uy_t30 - uy_ym1);
    duzdy_t30 = (uz_t30 - uz_ym1);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    duxdz_t30 = (ux_yp1 - ux_t30);
    duydz_t30 = (uy_yp1 - uy_t30);
    duzdz_t30 = (uz_yp1 - uz_t30);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    duxdz_t30 = (ux_ym1 - ux_t30);
    duydz_t30 = (uy_ym1 - uy_t30);
    duzdz_t30 = (uz_ym1 - uz_t30);
}


