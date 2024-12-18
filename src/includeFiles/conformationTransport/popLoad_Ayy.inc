if (tx == 0) { //w
    AyyNode[ 1] = ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
    AyyNode[ 7] = ghostInterface.Ayy_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
    AyyNode[ 9] = ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
    AyyNode[13] = ghostInterface.Ayy_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AyyNode[15] = ghostInterface.Ayy_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (tx == (BLOCK_NX - 1))
{ //e
    AyyNode[ 2] = ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
    AyyNode[ 8] = ghostInterface.Ayy_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AyyNode[10] = ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AyyNode[14] = ghostInterface.Ayy_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
    AyyNode[16] = ghostInterface.Ayy_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
}

if (ty == 0)
{ //s
    AyyNode[ 3] = ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
    AyyNode[ 7] = ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
    AyyNode[11] = ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
    AyyNode[14] = ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
    AyyNode[17] = ghostInterface.Ayy_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (ty == (BLOCK_NY - 1))
{ //n
    AyyNode[ 4] = ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
    AyyNode[ 8] = ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
    AyyNode[12] = ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AyyNode[13] = ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
    AyyNode[18] = ghostInterface.Ayy_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
}

if (tz == 0)
{ //b
    AyyNode[ 5] = ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
    AyyNode[ 9] = ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
    AyyNode[11] = ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
    AyyNode[16] = ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
    AyyNode[18] = ghostInterface.Ayy_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
}
else if (tz == (BLOCK_NZ - 1))
{ //f
    AyyNode[ 6] = ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
    AyyNode[10] = ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
    AyyNode[12] = ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
    AyyNode[15] = ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
    AyyNode[17] = ghostInterface.Ayy_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
}