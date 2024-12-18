if (tx == 0) { //w
    AzzNode[ 1] = ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
    AzzNode[ 7] = ghostInterface.Azz_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
    AzzNode[ 9] = ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
    AzzNode[13] = ghostInterface.Azz_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AzzNode[15] = ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (tx == (BLOCK_NX - 1))
{ //e
    AzzNode[ 2] = ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
    AzzNode[ 8] = ghostInterface.Azz_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AzzNode[10] = ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AzzNode[14] = ghostInterface.Azz_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
    AzzNode[16] = ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
}

if (ty == 0)
{ //s
    AzzNode[ 3] = ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
    AzzNode[ 7] = ghostInterface.Azz_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
    AzzNode[11] = ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
    AzzNode[14] = ghostInterface.Azz_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
    AzzNode[17] = ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (ty == (BLOCK_NY - 1))
{ //n
    AzzNode[ 4] = ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
    AzzNode[ 8] = ghostInterface.Azz_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
    AzzNode[12] = ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AzzNode[13] = ghostInterface.Azz_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
    AzzNode[18] = ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
}

if (tz == 0)
{ //b
    AzzNode[ 5] = ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
    AzzNode[ 9] = ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
    AzzNode[11] = ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
    AzzNode[16] = ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
    AzzNode[18] = ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
}
else if (tz == (BLOCK_NZ - 1))
{ //f
    AzzNode[ 6] = ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
    AzzNode[10] = ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
    AzzNode[12] = ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
    AzzNode[15] = ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
    AzzNode[17] = ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
}