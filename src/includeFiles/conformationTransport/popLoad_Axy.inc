if (tx == 0) { //w
    AxyNode[ 1] = ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
    AxyNode[ 7] = ghostInterface.Axy_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
    AxyNode[ 9] = ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
    AxyNode[13] = ghostInterface.Axy_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AxyNode[15] = ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (tx == (BLOCK_NX - 1))
{ //e
    AxyNode[ 2] = ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
    AxyNode[ 8] = ghostInterface.Axy_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AxyNode[10] = ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AxyNode[14] = ghostInterface.Axy_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
    AxyNode[16] = ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
}

if (ty == 0)
{ //s
    AxyNode[ 3] = ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
    AxyNode[ 7] = ghostInterface.Axy_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
    AxyNode[11] = ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
    AxyNode[14] = ghostInterface.Axy_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
    AxyNode[17] = ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (ty == (BLOCK_NY - 1))
{ //n
    AxyNode[ 4] = ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
    AxyNode[ 8] = ghostInterface.Axy_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
    AxyNode[12] = ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AxyNode[13] = ghostInterface.Axy_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
    AxyNode[18] = ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
}

if (tz == 0)
{ //b
    AxyNode[ 5] = ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
    AxyNode[ 9] = ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
    AxyNode[11] = ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
    AxyNode[16] = ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
    AxyNode[18] = ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
}
else if (tz == (BLOCK_NZ - 1))
{ //f
    AxyNode[ 6] = ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
    AxyNode[10] = ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
    AxyNode[12] = ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
    AxyNode[15] = ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
    AxyNode[17] = ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
}