if (tx == 0) {                                 
    sharedConf[idxConfBlock(tx - 1, ty, tz, 0)] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 0, bxm1, by, bz)];
    sharedConf[idxConfBlock(tx - 1, ty, tz, 1)] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 1, bxm1, by, bz)];
    sharedConf[idxConfBlock(tx - 1, ty, tz, 2)] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 2, bxm1, by, bz)];
    sharedConf[idxConfBlock(tx - 1, ty, tz, 3)] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 3, bxm1, by, bz)];
    sharedConf[idxConfBlock(tx - 1, ty, tz, 4)] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 4, bxm1, by, bz)];
    sharedConf[idxConfBlock(tx - 1, ty, tz, 5)] = ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 5, bxm1, by, bz)];
}
if (tx == BLOCK_NX - 1) {
    sharedConf[idxConfBlock(tx + 1, ty, tz, 0)] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 0, bxp1, by, bz)];
    sharedConf[idxConfBlock(tx + 1, ty, tz, 1)] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 1, bxp1, by, bz)];
    sharedConf[idxConfBlock(tx + 1, ty, tz, 2)] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 2, bxp1, by, bz)];
    sharedConf[idxConfBlock(tx + 1, ty, tz, 3)] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 3, bxp1, by, bz)];
    sharedConf[idxConfBlock(tx + 1, ty, tz, 4)] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 4, bxp1, by, bz)];
    sharedConf[idxConfBlock(tx + 1, ty, tz, 5)] = ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 5, bxp1, by, bz)];
}
if (ty == 0) {
    sharedConf[idxConfBlock(tx, ty - 1, tz, 0)] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 0, bx, bym1, bz)];
    sharedConf[idxConfBlock(tx, ty - 1, tz, 1)] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 1, bx, bym1, bz)];
    sharedConf[idxConfBlock(tx, ty - 1, tz, 2)] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 2, bx, bym1, bz)];
    sharedConf[idxConfBlock(tx, ty - 1, tz, 3)] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 3, bx, bym1, bz)];
    sharedConf[idxConfBlock(tx, ty - 1, tz, 4)] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 4, bx, bym1, bz)];
    sharedConf[idxConfBlock(tx, ty - 1, tz, 5)] = ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 5, bx, bym1, bz)];
}
if (ty == BLOCK_NY - 1) {
    sharedConf[idxConfBlock(tx, ty + 1, tz, 0)] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, ty, 0, bx, byp1, bz)];
    sharedConf[idxConfBlock(tx, ty + 1, tz, 1)] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, ty, 1, bx, byp1, bz)];
    sharedConf[idxConfBlock(tx, ty + 1, tz, 2)] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, ty, 2, bx, byp1, bz)];
    sharedConf[idxConfBlock(tx, ty + 1, tz, 3)] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, ty, 3, bx, byp1, bz)];
    sharedConf[idxConfBlock(tx, ty + 1, tz, 4)] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, ty, 4, bx, byp1, bz)];
    sharedConf[idxConfBlock(tx, ty + 1, tz, 5)] = ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, ty, 5, bx, byp1, bz)];
}
if (tz == 0) {
    sharedConf[idxConfBlock(tx, ty, tz - 1, 0)] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 0, bx, by, bzm1)];
    sharedConf[idxConfBlock(tx, ty, tz - 1, 1)] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 1, bx, by, bzm1)];
    sharedConf[idxConfBlock(tx, ty, tz - 1, 2)] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 2, bx, by, bzm1)];
    sharedConf[idxConfBlock(tx, ty, tz - 1, 3)] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 3, bx, by, bzm1)];
    sharedConf[idxConfBlock(tx, ty, tz - 1, 4)] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 4, bx, by, bzm1)];
    sharedConf[idxConfBlock(tx, ty, tz - 1, 5)] = ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 5, bx, by, bzm1)];
}
if (tz == BLOCK_NZ - 1) {
    sharedConf[idxConfBlock(tx, ty, tz + 1, 0)] = ghostInterface.conf_fGhost.Z_0[g_idxConfX(tx, ty, 0, bx, by, bzp1)];
    sharedConf[idxConfBlock(tx, ty, tz + 1, 1)] = ghostInterface.conf_fGhost.Z_0[g_idxConfX(tx, ty, 1, bx, by, bzp1)];
    sharedConf[idxConfBlock(tx, ty, tz + 1, 2)] = ghostInterface.conf_fGhost.Z_0[g_idxConfX(tx, ty, 2, bx, by, bzp1)];
    sharedConf[idxConfBlock(tx, ty, tz + 1, 3)] = ghostInterface.conf_fGhost.Z_0[g_idxConfX(tx, ty, 3, bx, by, bzp1)];
    sharedConf[idxConfBlock(tx, ty, tz + 1, 4)] = ghostInterface.conf_fGhost.Z_0[g_idxConfX(tx, ty, 4, bx, by, bzp1)];
    sharedConf[idxConfBlock(tx, ty, tz + 1, 5)] = ghostInterface.conf_fGhost.Z_0[g_idxConfX(tx, ty, 5, bx, by, bzp1)];
}