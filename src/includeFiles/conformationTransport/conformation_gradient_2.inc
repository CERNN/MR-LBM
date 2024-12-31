//TODO FOR SOME REASON ISNT WORKING, SOMETHING WRONG WITH INDEXING.

shared_index = 6 * hx + 
               6 * haloNx * hy + 
               6 * haloNx * haloNy * hz;

const int sharedIndexAxx = shared_index + 0;
const int sharedIndexAxy = shared_index + 1;
const int sharedIndexAxz = shared_index + 2;
const int sharedIndexAyy = shared_index + 3;
const int sharedIndexAyz = shared_index + 4;
const int sharedIndexAzz = shared_index + 5;

s_pop[sharedIndexAxx] = AxxVar;
s_pop[sharedIndexAxy] = AxyVar;
s_pop[sharedIndexAxz] = AxzVar;
s_pop[sharedIndexAyy] = AyyVar;
s_pop[sharedIndexAyz] = AyzVar;
s_pop[sharedIndexAzz] = AzzVar;

//load halo
if (tx == 0) {
    s_pop[sharedIndexAxx - 6] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 0, bxm1, by, bz)];
    s_pop[sharedIndexAxy - 6] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 1, bxm1, by, bz)];
    s_pop[sharedIndexAxz - 6] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 2, bxm1, by, bz)];
    s_pop[sharedIndexAyy - 6] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 3, bxm1, by, bz)];
    s_pop[sharedIndexAyz - 6] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 4, bxm1, by, bz)];
    s_pop[sharedIndexAzz - 6] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 5, bxm1, by, bz)];
}
if (tx == BLOCK_NX - 1) {
    s_pop[sharedIndexAxx + 6] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 0, bxp1, by, bz)];
    s_pop[sharedIndexAxy + 6] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 1, bxp1, by, bz)];
    s_pop[sharedIndexAxz + 6] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 2, bxp1, by, bz)];
    s_pop[sharedIndexAyy + 6] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 3, bxp1, by, bz)];
    s_pop[sharedIndexAyz + 6] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 4, bxp1, by, bz)];
    s_pop[sharedIndexAzz + 6] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 5, bxp1, by, bz)];
}

if (ty == 0) {
    s_pop[sharedIndexAxx - 6*haloNx] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 0, bx, bym1, bz)];
    s_pop[sharedIndexAxy - 6*haloNx] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 1, bx, bym1, bz)];
    s_pop[sharedIndexAxz - 6*haloNx] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 2, bx, bym1, bz)];
    s_pop[sharedIndexAyy - 6*haloNx] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 3, bx, bym1, bz)];
    s_pop[sharedIndexAyz - 6*haloNx] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 4, bx, bym1, bz)];
    s_pop[sharedIndexAzz - 6*haloNx] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 5, bx, bym1, bz)];
}
if (ty == BLOCK_NY - 1) {
    s_pop[sharedIndexAxx + 6*haloNx] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 0, bx, byp1, bz)];
    s_pop[sharedIndexAxy + 6*haloNx] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 1, bx, byp1, bz)];
    s_pop[sharedIndexAxz + 6*haloNx] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 2, bx, byp1, bz)];
    s_pop[sharedIndexAyy + 6*haloNx] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 3, bx, byp1, bz)];
    s_pop[sharedIndexAyz + 6*haloNx] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 4, bx, byp1, bz)];
    s_pop[sharedIndexAzz + 6*haloNx] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 5, bx, byp1, bz)];
}

if (tz == 0) {
    s_pop[sharedIndexAxx - 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 0, bx, by, bzm1)];
    s_pop[sharedIndexAxy - 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 1, bx, by, bzm1)];
    s_pop[sharedIndexAxz - 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 2, bx, by, bzm1)];
    s_pop[sharedIndexAyy - 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 3, bx, by, bzm1)];
    s_pop[sharedIndexAyz - 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 4, bx, by, bzm1)];
    s_pop[sharedIndexAzz - 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(ty, tz, 5, bx, by, bzm1)];
}
if (tz == BLOCK_NZ - 1) {
    s_pop[sharedIndexAxx + 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 0, bx, by, bzp1)];
    s_pop[sharedIndexAxy + 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 1, bx, by, bzp1)];
    s_pop[sharedIndexAxz + 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 2, bx, by, bzp1)];
    s_pop[sharedIndexAyy + 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 3, bx, by, bzp1)];
    s_pop[sharedIndexAyz + 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 4, bx, by, bzp1)];
    s_pop[sharedIndexAzz + 6*haloNx*haloNy] = ghostInterface.conf_fGhost.Z_0[g_idxConfZ(ty, tz, 5, bx, by, bzp1)];
}

dfloat Axx_xm1 = s_pop[sharedIndexAxx - 6];
dfloat Axy_xm1 = s_pop[sharedIndexAxy - 6];
dfloat Axz_xm1 = s_pop[sharedIndexAxz - 6];
//dfloat Ayy_xm1 = s_pop[sharedIndexAyy - 6];
//dfloat Ayz_xm1 = s_pop[sharedIndexAyz - 6];
//dfloat Azz_xm1 = s_pop[sharedIndexAzz - 6];

dfloat Axx_xp1 = s_pop[sharedIndexAxx + 6];
dfloat Axy_xp1 = s_pop[sharedIndexAxy + 6];
dfloat Axz_xp1 = s_pop[sharedIndexAxz + 6];
//dfloat Ayy_xp1 = s_pop[sharedIndexAyy + 6];
//dfloat Ayz_xp1 = s_pop[sharedIndexAyz + 6];
//dfloat Azz_xp1 = s_pop[sharedIndexAzz + 6];

//dfloat Axx_ym1 = s_pop[sharedIndexAxx - 6*haloNx];
dfloat Axy_ym1 = s_pop[sharedIndexAxy - 6*haloNx];
//dfloat Axz_ym1 = s_pop[sharedIndexAxz - 6*haloNx];
dfloat Ayy_ym1 = s_pop[sharedIndexAyy - 6*haloNx];
dfloat Ayz_ym1 = s_pop[sharedIndexAyz - 6*haloNx];
//dfloat Azz_ym1 = s_pop[sharedIndexAzz - 6*haloNx];

//dfloat Axx_yp1 = s_pop[sharedIndexAxx + 6*haloNx];
dfloat Axy_yp1 = s_pop[sharedIndexAxy + 6*haloNx];
//dfloat Axz_yp1 = s_pop[sharedIndexAxz + 6*haloNx];
dfloat Ayy_yp1 = s_pop[sharedIndexAyy + 6*haloNx];
dfloat Ayz_yp1 = s_pop[sharedIndexAyz + 6*haloNx];
//dfloat Azz_yp1 = s_pop[sharedIndexAzz + 6*haloNx];

//dfloat Axx_zm1 = s_pop[sharedIndexAxx - 6*haloNx*haloNy];
//dfloat Axy_zm1 = s_pop[sharedIndexAxy - 6*haloNx*haloNy];
dfloat Axz_zm1 = s_pop[sharedIndexAxz - 6*haloNx*haloNy];
//dfloat Ayy_zm1 = s_pop[sharedIndexAyy - 6*haloNx*haloNy];
dfloat Ayz_zm1 = s_pop[sharedIndexAyz - 6*haloNx*haloNy];
dfloat Azz_zm1 = s_pop[sharedIndexAzz - 6*haloNx*haloNy];

//dfloat Axx_zp1 = s_pop[sharedIndexAxx + 6*haloNx*haloNy];
//dfloat Axy_zp1 = s_pop[sharedIndexAxy + 6*haloNx*haloNy];
dfloat Axz_zp1 = s_pop[sharedIndexAxz + 6*haloNx*haloNy];
//dfloat Ayy_zp1 = s_pop[sharedIndexAyy + 6*haloNx*haloNy];
dfloat Ayz_zp1 = s_pop[sharedIndexAyz + 6*haloNx*haloNy];
dfloat Azz_zp1 = s_pop[sharedIndexAzz + 6*haloNx*haloNy];

// Compute derivatives
dfloat dAxxdx = (Axx_xp1 - Axx_xm1) / 2;
dfloat dAxydy = (Axy_yp1 - Axy_ym1) / 2;
dfloat dAxzdz = (Axz_zp1 - Axz_zm1) / 2;

dfloat dAxydx = (Axy_xp1 - Axy_xm1) / 2;
dfloat dAyydy = (Ayy_yp1 - Ayy_ym1) / 2;
dfloat dAyzdz = (Ayz_zp1 - Ayz_zm1) / 2;

dfloat dAxzdx = (Axz_xp1 - Axz_xm1) / 2;
dfloat dAyzdy = (Ayz_yp1 - Ayz_ym1) / 2;
dfloat dAzzdz = (Azz_zp1 - Azz_zm1) / 2;


