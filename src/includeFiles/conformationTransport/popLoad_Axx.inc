if (tx == 0) { //w
    AxxNode[ 1] = ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
    AxxNode[ 7] = ghostInterface.Axx_fGhost.X_1[g_idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
    AxxNode[ 9] = ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
    AxxNode[13] = ghostInterface.Axx_fGhost.X_1[g_idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AxxNode[15] = ghostInterface.Axx_fGhost.X_1[g_idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (tx == (BLOCK_NX - 1))
{ //e
    AxxNode[ 2] = ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
    AxxNode[ 8] = ghostInterface.Axx_fGhost.X_0[g_idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
    AxxNode[10] = ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AxxNode[14] = ghostInterface.Axx_fGhost.X_0[g_idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
    AxxNode[16] = ghostInterface.Axx_fGhost.X_0[g_idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
}

if (ty == 0)
{ //s
    AxxNode[ 3] = ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
    AxxNode[ 7] = ghostInterface.Axx_fGhost.Y_1[g_idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
    AxxNode[11] = ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
    AxxNode[14] = ghostInterface.Axx_fGhost.Y_1[g_idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
    AxxNode[17] = ghostInterface.Axx_fGhost.Y_1[g_idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
}
else if (ty == (BLOCK_NY - 1))
{ //n
    AxxNode[ 4] = ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
    AxxNode[ 8] = ghostInterface.Axx_fGhost.Y_0[g_idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
    AxxNode[12] = ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
    AxxNode[13] = ghostInterface.Axx_fGhost.Y_0[g_idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
    AxxNode[18] = ghostInterface.Axx_fGhost.Y_0[g_idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
}

if (tz == 0)
{ //b
    AxxNode[ 5] = ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
    AxxNode[ 9] = ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
    AxxNode[11] = ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
    AxxNode[16] = ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
    AxxNode[18] = ghostInterface.Axx_fGhost.Z_1[g_idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
}
else if (tz == (BLOCK_NZ - 1))
{ //f
    AxxNode[ 6] = ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
    AxxNode[10] = ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
    AxxNode[12] = ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
    AxxNode[15] = ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
    AxxNode[17] = ghostInterface.Axx_fGhost.Z_0[g_idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
}