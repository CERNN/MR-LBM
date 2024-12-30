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


dfloat duxdx_t30 = (ux_xp1 - ux_xm1) / 2;
dfloat duydx_t30 = (uy_xp1 - uy_xm1) / 2;
dfloat duzdx_t30 = (uz_xp1 - uz_xm1) / 2;

dfloat duxdy_t30 = (ux_yp1 - ux_ym1) / 2;
dfloat duydy_t30 = (uy_yp1 - uy_ym1) / 2;
dfloat duzdy_t30 = (uz_yp1 - uz_ym1) / 2;

dfloat duxdz_t30 = (ux_zp1 - ux_zm1) / 2;
dfloat duydz_t30 = (uy_zp1 - uy_zm1) / 2;
dfloat duzdz_t30 = (uz_zp1 - uz_zm1) / 2;


/*
const int haloNx = BLOCK_NX + 2;
const int haloNy = BLOCK_NY + 2;
const int haloNz = BLOCK_NZ + 2;

const int sharedIndexUx = (tx + 1) + (ty + 1) * haloNx + (tz + 1) * haloNx * haloNy;
const int sharedIndexUy = sharedIndexUx + haloNx * haloNy * haloNz;
const int sharedIndexUz = sharedIndexUy + haloNx * haloNy * haloNz;

s_pop[sharedIndexUx] = ux_t30;
s_pop[sharedIndexUy] = uy_t30;
s_pop[sharedIndexUz] = uz_t30;

// Load halo regions
if (tx == 0) {
    s_pop[sharedIndexUx - 1] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 0, bxm1, by, bz)];
    s_pop[sharedIndexUy - 1] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 1, bxm1, by, bz)];
    s_pop[sharedIndexUz - 1] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 2, bxm1, by, bz)];
}
if (tx == BLOCK_NX - 1) {
    s_pop[sharedIndexUx + 1] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 0, bxp1, by, bz)];
    s_pop[sharedIndexUy + 1] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 1, bxp1, by, bz)];
    s_pop[sharedIndexUz + 1] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 2, bxp1, by, bz)];
}

if (ty == 0) {
    s_pop[sharedIndexUx - haloNx] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 0, bx, bym1, bz)];
    s_pop[sharedIndexUy - haloNx] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 1, bx, bym1, bz)];
    s_pop[sharedIndexUz - haloNx] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 2, bx, bym1, bz)];
}
if (ty == BLOCK_NY - 1) {
    s_pop[sharedIndexUx + haloNx] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 0, bx, byp1, bz)];
    s_pop[sharedIndexUy + haloNx] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 1, bx, byp1, bz)];
    s_pop[sharedIndexUz + haloNx] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 2, bx, byp1, bz)];
}

if (tz == 0) {
    s_pop[sharedIndexUx - haloNx * haloNy] = ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 0, bx, by, bzm1)];
    s_pop[sharedIndexUy - haloNx * haloNy] = ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 1, bx, by, bzm1)];
    s_pop[sharedIndexUz - haloNx * haloNy] = ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 2, bx, by, bzm1)];
}
if (tz == BLOCK_NZ - 1) {
    s_pop[sharedIndexUx + haloNx * haloNy] = ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 0, bx, by, bzp1)];
    s_pop[sharedIndexUy + haloNx * haloNy] = ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 1, bx, by, bzp1)];
    s_pop[sharedIndexUz + haloNx * haloNy] = ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 2, bx, by, bzp1)];
}

__syncthreads();

// Compute gradients
dfloat duxdx_t30 = (s_pop[sharedIndexUx + 1] - s_pop[sharedIndexUx - 1]) / 2.0;
dfloat duxdy_t30 = (s_pop[sharedIndexUx + haloNx] - s_pop[sharedIndexUx - haloNx]) / 2.0;
dfloat duxdz_t30 = (s_pop[sharedIndexUx + haloNx * haloNy] - s_pop[sharedIndexUx - haloNx * haloNy]) / 2.0;

dfloat duydx_t30 = (s_pop[sharedIndexUy + 1] - s_pop[sharedIndexUy - 1]) / 2.0;
dfloat duydy_t30 = (s_pop[sharedIndexUy + haloNx] - s_pop[sharedIndexUy - haloNx]) / 2.0;
dfloat duydz_t30 = (s_pop[sharedIndexUy + haloNx * haloNy] - s_pop[sharedIndexUy - haloNx * haloNy]) / 2.0;

dfloat duzdx_t30 = (s_pop[sharedIndexUz + 1] - s_pop[sharedIndexUz - 1]) / 2.0;
dfloat duzdy_t30 = (s_pop[sharedIndexUz + haloNx] - s_pop[sharedIndexUz - haloNx]) / 2.0;
dfloat duzdz_t30 = (s_pop[sharedIndexUz + haloNx * haloNy] - s_pop[sharedIndexUz - haloNx * haloNy]) / 2.0;
*/


/*
s_pop[idxVelBlock(tx, ty, tz, 0)] = ux_t30;
s_pop[idxVelBlock(tx, ty, tz, 1)] = uy_t30;
s_pop[idxVelBlock(tx, ty, tz, 2)] = uz_t30;

__syncthreads();

dfloat ux_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 0)];
dfloat uy_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 1)];
dfloat uz_xm1 = s_pop[idxVelBlock(txm1, ty, tz, 2)];

dfloat ux_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 0)];
dfloat uy_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 1)];
dfloat uz_xp1 = s_pop[idxVelBlock(txp1, ty, tz, 2)];

dfloat ux_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 0)];
dfloat uy_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 1)];
dfloat uz_ym1 = s_pop[idxVelBlock(tx, tym1, tz, 2)];

dfloat ux_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 0)];
dfloat uy_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 1)];
dfloat uz_yp1 = s_pop[idxVelBlock(tx, typ1, tz, 2)];

dfloat ux_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 0)];
dfloat uy_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 1)];
dfloat uz_zm1 = s_pop[idxVelBlock(tx, ty, tzm1, 2)];

dfloat ux_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 0)];
dfloat uy_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 1)];
dfloat uz_zp1 = s_pop[idxVelBlock(tx, ty, tzp1, 2)];



// Compute unrolled gradient calculations
//FINITE CENTRAL DIFFERENCE ORDER 1, otherwise i would have to store a halo with width of 2 in shared memory
dfloat duxdx_t30 = (ux_xp1 - ux_xm1) / 2;
dfloat duydx_t30 = (uy_xp1 - uy_xm1) / 2;
dfloat duzdx_t30 = (uz_xp1 - uz_xm1) / 2;

dfloat duxdy_t30 = (ux_yp1 - ux_ym1) / 2;
dfloat duydy_t30 = (uy_yp1 - uy_ym1) / 2;
dfloat duzdy_t30 = (uz_yp1 - uz_ym1) / 2;

dfloat duxdz_t30 = (ux_zp1 - ux_zm1) / 2;
dfloat duydz_t30 = (uy_zp1 - uy_zm1) / 2;
dfloat duzdz_t30 = (uz_zp1 - uz_zm1) / 2;

if (tx == 0) {   
    duxdx_t30 = (ux_xp1 - ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 0, bxm1, by, bz)]) / 2;
    duydx_t30 = (uy_xp1 - ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 1, bxm1, by, bz)]) / 2;
    duzdx_t30 = (uz_xp1 - ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 2, bxm1, by, bz)]) / 2;
}
if (tx == BLOCK_NX - 1) {
    duxdx_t30 = (ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 0, bxp1, by, bz)] - ux_xm1) / 2;
    duydx_t30 = (ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 1, bxp1, by, bz)] - uy_xm1) / 2;
    duzdx_t30 = (ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 2, bxp1, by, bz)] - uz_xm1) / 2;
}

if (ty == 0) {   
    duxdy_t30 = (ux_yp1 - ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 0, bx, bym1, bz)]) / 2;
    duydy_t30 = (uy_yp1 - ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 1, bx, bym1, bz)]) / 2;
    duzdy_t30 = (uz_yp1 - ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 2, bx, bym1, bz)]) / 2;
}
if (ty == BLOCK_NY - 1) {
    duxdy_t30 = (ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 0, bx, byp1, bz)] - ux_ym1) / 2;
    duydy_t30 = (ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 1, bx, byp1, bz)] - uy_ym1) / 2;
    duzdy_t30 = (ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 2, bx, byp1, bz)] - uz_ym1) / 2;
}

if (tz == 0) {   
    duxdz_t30 = (ux_zp1 - ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 0, bx, by, bzm1)]) / 2;
    duydz_t30 = (uy_zp1 - ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 1, bx, by, bzm1)]) / 2;
    duzdz_t30 = (uz_zp1 - ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 2, bx, by, bzm1)]) / 2;
}
if (tz == BLOCK_NZ - 1) {
    duxdz_t30 = (ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 0, bx, by, bzp1)] - ux_zm1) / 2;
    duydz_t30 = (ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 1, bx, by, bzp1)] - uy_zm1) / 2;
    duzdz_t30 = (ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 2, bx, by, bzp1)] - uz_zm1) / 2;
}
*/

//if(x == 32 && z == 32)
//    printf("y: %d duydy_t30 %f uy_p1 %f uy %f uy_m1 %f  \n",y,duydy_t30,s_pop[idxVelBlock(tx, ty + 1, tz, 3)],s_pop[idxVelBlock(tx, ty, tz, 3)],s_pop[idxVelBlock(tx, ty - 1, tz, 3)]);

//OVERWRITE FOR NEAR WALL NODES
/*
if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    duxdx_t30 = (s_pop[idxVelBlock(tx + 1, ty, tz, 0)] - s_pop[idxVelBlock(tx, ty, tz, 0)]);
    duydx_t30 = (s_pop[idxVelBlock(tx + 1, ty, tz, 1)] - s_pop[idxVelBlock(tx, ty, tz, 1)]);
    duzdx_t30 = (s_pop[idxVelBlock(tx + 1, ty, tz, 2)] - s_pop[idxVelBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    duxdx_t30 = (s_pop[idxVelBlock(tx, ty, tz, 0)] - s_pop[idxVelBlock(tx - 1, ty, tz, 0)]);
    duydx_t30 = (s_pop[idxVelBlock(tx, ty, tz, 1)] - s_pop[idxVelBlock(tx - 1, ty, tz, 1)]);
    duzdx_t30 = (s_pop[idxVelBlock(tx, ty, tz, 2)] - s_pop[idxVelBlock(tx - 1, ty, tz, 2)]);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    duxdy_t30 = (s_pop[idxVelBlock(tx, ty + 1, tz, 0)] - s_pop[idxVelBlock(tx, ty, tz, 0)]);
    duydy_t30 = (s_pop[idxVelBlock(tx, ty + 1, tz, 1)] - s_pop[idxVelBlock(tx, ty, tz, 1)]);
    duzdy_t30 = (s_pop[idxVelBlock(tx, ty + 1, tz, 2)] - s_pop[idxVelBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    duxdy_t30 = (s_pop[idxVelBlock(tx, ty, tz, 0)] - s_pop[idxVelBlock(tx, ty - 1, tz, 0)]);
    duydy_t30 = (s_pop[idxVelBlock(tx, ty, tz, 1)] - s_pop[idxVelBlock(tx, ty - 1, tz, 1)]);
    duzdy_t30 = (s_pop[idxVelBlock(tx, ty, tz, 2)] - s_pop[idxVelBlock(tx, ty - 1, tz, 2)]);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    duxdz_t30 = (s_pop[idxVelBlock(tx, ty, tz + 1, 0)] - s_pop[idxVelBlock(tx, ty, tz, 0)]);
    duydz_t30 = (s_pop[idxVelBlock(tx, ty, tz + 1, 1)] - s_pop[idxVelBlock(tx, ty, tz, 1)]);
    duzdz_t30 = (s_pop[idxVelBlock(tx, ty, tz + 1, 2)] - s_pop[idxVelBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    duxdz_t30 = (s_pop[idxVelBlock(tx, ty, tz, 0)] - s_pop[idxVelBlock(tx, ty, tz - 1, 0)]);
    duydz_t30 = (s_pop[idxVelBlock(tx, ty, tz, 1)] - s_pop[idxVelBlock(tx, ty, tz - 1, 1)]);
    duzdz_t30 = (s_pop[idxVelBlock(tx, ty, tz, 2)] - s_pop[idxVelBlock(tx, ty, tz - 1, 2)]);
}
*/
//if(x == 4 && y == 4 && z == 4 )
//    printf("step %d grad %f \n",step,duzdy_t30/3.0);