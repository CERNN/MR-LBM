
const int haloNx = BLOCK_NX + 2*HALO_SIZE;
const int haloNy = BLOCK_NY + 2*HALO_SIZE;
const int haloNz = BLOCK_NZ + 2*HALO_SIZE;

int hx = tx + HALO_SIZE;
int hy = ty + HALO_SIZE;
int hz = tz + HALO_SIZE;

int shared_index = 3 * hx + 
                   3 * haloNx * hy + 
                   3 * haloNx * haloNy * hz;

const int sharedIndexUx = shared_index + 0;
const int sharedIndexUy = shared_index + 1;
const int sharedIndexUz = shared_index + 2;

s_pop[sharedIndexUx] = ux_t30;
s_pop[sharedIndexUy] = uy_t30;
s_pop[sharedIndexUz] = uz_t30;

//load halo
if (tx == 0) {
    s_pop[sharedIndexUx - 3] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 0, bxm1, by, bz)];
    s_pop[sharedIndexUy - 3] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 1, bxm1, by, bz)];
    s_pop[sharedIndexUz - 3] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 2, bxm1, by, bz)];
}
if (tx == BLOCK_NX - 1) {
    s_pop[sharedIndexUx + 3] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 0, bxp1, by, bz)];
    s_pop[sharedIndexUy + 3] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 1, bxp1, by, bz)];
    s_pop[sharedIndexUz + 3] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 2, bxp1, by, bz)];
}

if (ty == 0) {
    s_pop[sharedIndexUx - 3*haloNx] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 0, bx, bym1, bz)];
    s_pop[sharedIndexUy - 3*haloNx] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 1, bx, bym1, bz)];
    s_pop[sharedIndexUz - 3*haloNx] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 2, bx, bym1, bz)];
}
if (ty == BLOCK_NY - 1) {
    s_pop[sharedIndexUx + 3*haloNx] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 0, bx, byp1, bz)];
    s_pop[sharedIndexUy + 3*haloNx] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 1, bx, byp1, bz)];
    s_pop[sharedIndexUz + 3*haloNx] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 2, bx, byp1, bz)];
}

if (tz == 0) {
    s_pop[sharedIndexUx - 3*haloNx*haloNy] = ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 0, bx, by, bzm1)];
    s_pop[sharedIndexUy - 3*haloNx*haloNy] = ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 1, bx, by, bzm1)];
    s_pop[sharedIndexUz - 3*haloNx*haloNy] = ghostInterface.f_uGhost.Z_1[g_idxUZ(ty, tz, 2, bx, by, bzm1)];
}
if (tz == BLOCK_NZ - 1) {
    s_pop[sharedIndexUx + 3*haloNx*haloNy] = ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 0, bx, by, bzp1)];
    s_pop[sharedIndexUy + 3*haloNx*haloNy] = ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 1, bx, by, bzp1)];
    s_pop[sharedIndexUz + 3*haloNx*haloNy] = ghostInterface.f_uGhost.Z_0[g_idxUZ(ty, tz, 2, bx, by, bzp1)];
}

__syncthreads();

dfloat ux_xm1 = s_pop[sharedIndexUx - 3];
dfloat uy_xm1 = s_pop[sharedIndexUy - 3];
dfloat uz_xm1 = s_pop[sharedIndexUz - 3];

dfloat ux_xp1 = s_pop[sharedIndexUx + 3];
dfloat uy_xp1 = s_pop[sharedIndexUy + 3];
dfloat uz_xp1 = s_pop[sharedIndexUz + 3];

dfloat ux_ym1 = s_pop[sharedIndexUx - 3*haloNx];
dfloat uy_ym1 = s_pop[sharedIndexUy - 3*haloNx];
dfloat uz_ym1 = s_pop[sharedIndexUz - 3*haloNx];

dfloat ux_yp1 = s_pop[sharedIndexUx + 3*haloNx];
dfloat uy_yp1 = s_pop[sharedIndexUy + 3*haloNx];
dfloat uz_yp1 = s_pop[sharedIndexUz + 3*haloNx];

dfloat ux_zm1 = s_pop[sharedIndexUx - 3*haloNx*haloNy];
dfloat uy_zm1 = s_pop[sharedIndexUy - 3*haloNx*haloNy];
dfloat uz_zm1 = s_pop[sharedIndexUz - 3*haloNx*haloNy];

dfloat ux_zp1 = s_pop[sharedIndexUx + 3*haloNx*haloNy];
dfloat uy_zp1 = s_pop[sharedIndexUy + 3*haloNx*haloNy];
dfloat uz_zp1 = s_pop[sharedIndexUz + 3*haloNx*haloNy];

// Compute unrolled gradient calculations
dfloat duxdx_t30 = (ux_xp1 - ux_xm1) / 2.0;
dfloat duydx_t30 = (uy_xp1 - uy_xm1) / 2.0;
dfloat duzdx_t30 = (uz_xp1 - uz_xm1) / 2.0;

dfloat duxdy_t30 = (ux_yp1 - ux_ym1) / 2.0;
dfloat duydy_t30 = (uy_yp1 - uy_ym1) / 2.0;
dfloat duzdy_t30 = (uz_yp1 - uz_ym1) / 2.0;

dfloat duxdz_t30 = (ux_zp1 - ux_zm1) / 2.0;
dfloat duydz_t30 = (uy_zp1 - uy_zm1) / 2.0;
dfloat duzdz_t30 = (uz_zp1 - uz_zm1) / 2.0;
