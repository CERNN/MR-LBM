if (tx == 0) {                                 
    s_pop[idxVelBlock(tx - 1, ty, tz, 0)] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 0, bxm1, by, bz)];
    s_pop[idxVelBlock(tx - 1, ty, tz, 1)] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 1, bxm1, by, bz)];
    s_pop[idxVelBlock(tx - 1, ty, tz, 2)] = ghostInterface.f_uGhost.X_1[g_idxUX(ty, tz, 2, bxm1, by, bz)];
}
if (tx == BLOCK_NX - 1) {
    s_pop[idxVelBlock(tx + 1, ty, tz, 0)] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 0, bxp1, by, bz)];
    s_pop[idxVelBlock(tx + 1, ty, tz, 1)] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 1, bxp1, by, bz)];
    s_pop[idxVelBlock(tx + 1, ty, tz, 2)] = ghostInterface.f_uGhost.X_0[g_idxUX(ty, tz, 2, bxp1, by, bz)];
}
if (ty == 0) {
    s_pop[idxVelBlock(tx, ty - 1, tz, 0)] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 0, bx, bym1, bz)];
    s_pop[idxVelBlock(tx, ty - 1, tz, 1)] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 1, bx, bym1, bz)];
    s_pop[idxVelBlock(tx, ty - 1, tz, 2)] = ghostInterface.f_uGhost.Y_1[g_idxUY(tx, tz, 2, bx, bym1, bz)];
}
if (ty == BLOCK_NY - 1) {
    s_pop[idxVelBlock(tx, ty + 1, tz, 0)] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 0, bx, byp1, bz)];
    s_pop[idxVelBlock(tx, ty + 1, tz, 1)] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 1, bx, byp1, bz)];
    s_pop[idxVelBlock(tx, ty + 1, tz, 2)] = ghostInterface.f_uGhost.Y_0[g_idxUY(tx, tz, 2, bx, byp1, bz)];
}
if (tz == 0) {
    s_pop[idxVelBlock(tx, ty, tz - 1, 0)] = ghostInterface.f_uGhost.Z_1[g_idxUZ(tx, ty, 0, bx, by, bzm1)];
    s_pop[idxVelBlock(tx, ty, tz - 1, 1)] = ghostInterface.f_uGhost.Z_1[g_idxUZ(tx, ty, 1, bx, by, bzm1)];
    s_pop[idxVelBlock(tx, ty, tz - 1, 2)] = ghostInterface.f_uGhost.Z_1[g_idxUZ(tx, ty, 2, bx, by, bzm1)];
}
if (tz == BLOCK_NZ - 1) {
    s_pop[idxVelBlock(tx, ty, tz + 1, 0)] = ghostInterface.f_uGhost.Z_0[g_idxUZ(tx, ty, 0, bx, by, bzp1)];
    s_pop[idxVelBlock(tx, ty, tz + 1, 1)] = ghostInterface.f_uGhost.Z_0[g_idxUZ(tx, ty, 1, bx, by, bzp1)];
    s_pop[idxVelBlock(tx, ty, tz + 1, 2)] = ghostInterface.f_uGhost.Z_0[g_idxUZ(tx, ty, 2, bx, by, bzp1)];
}