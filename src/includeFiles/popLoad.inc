
if (tx == 0) { //w
        pop[ 1] = ghostInterface.fGhost.X_1[idxPopX(ty  , tz, 0, bxm1, by                                       , bz)];
        pop[ 7] = ghostInterface.fGhost.X_1[idxPopX(tym1, tz, 1, bxm1, ((ty == 0) ? bym1 : by)                  , bz)];
        pop[ 9] = ghostInterface.fGhost.X_1[idxPopX(ty, tzm1, 2, bxm1, by                                       , ((tz == 0) ? bzm1 : bz))];
        pop[13] = ghostInterface.fGhost.X_1[idxPopX(typ1, tz, 3, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        pop[15] = ghostInterface.fGhost.X_1[idxPopX(ty, tzp1, 4, bxm1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #ifdef D3Q27
        pop[19] = ghostInterface.fGhost.X_1[idxPopX(tym1, tzm1, 5, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
        pop[21] = ghostInterface.fGhost.X_1[idxPopX(tym1, tzp1, 6, bxm1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[23] = ghostInterface.fGhost.X_1[idxPopX(typ1, tzm1, 7, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
        pop[26] = ghostInterface.fGhost.X_1[idxPopX(typ1, tzp1, 8, bxm1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #endif //D3Q27
    }
else if (tx == (BLOCK_NX - 1))
    { //e
        pop[ 2] = ghostInterface.fGhost.X_0[idxPopX(ty  , tz, 0, bxp1, by                                       , bz)];
        pop[ 8] = ghostInterface.fGhost.X_0[idxPopX(typ1, tz, 1, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)     , bz)];
        pop[10] = ghostInterface.fGhost.X_0[idxPopX(ty, tzp1, 2, bxp1, by                                       , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[14] = ghostInterface.fGhost.X_0[idxPopX(tym1, tz, 3, bxp1, ((ty == 0) ? bym1 : by)                  , bz)];
        pop[16] = ghostInterface.fGhost.X_0[idxPopX(ty, tzm1, 4, bxp1, by                                       , ((tz == 0) ? bzm1 : bz))];
        #ifdef D3Q27
        pop[20] = ghostInterface.fGhost.X_0[idxPopX(typ1, tzp1, 5, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[22] = ghostInterface.fGhost.X_0[idxPopX(typ1, tzm1, 6, bxp1, ((ty == (BLOCK_NY - 1)) ? byp1 : by)   , ((tz == 0) ? bzm1 : bz))];
        pop[24] = ghostInterface.fGhost.X_0[idxPopX(tym1, tzp1, 7, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[25] = ghostInterface.fGhost.X_0[idxPopX(tym1, tzm1, 8, bxp1, ((ty == 0) ? bym1 : by)                , ((tz == 0) ? bzm1 : bz))];
        #endif //D3Q27
    }

if (ty == 0)
    { //s
        pop[ 3] = ghostInterface.fGhost.Y_1[idxPopY(tx  , tz, 0, bx                                     , bym1, bz)];
        pop[ 7] = ghostInterface.fGhost.Y_1[idxPopY(txm1, tz, 1, ((tx == 0) ? bxm1 : bx)                , bym1, bz)];
        pop[11] = ghostInterface.fGhost.Y_1[idxPopY(tx, tzm1, 2, bx                                     , bym1, ((tz == 0) ? bzm1 : bz))];
        pop[14] = ghostInterface.fGhost.Y_1[idxPopY(txp1, tz, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , bym1, bz)];
        pop[17] = ghostInterface.fGhost.Y_1[idxPopY(tx, tzp1, 4, bx                                     , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #ifdef D3Q27
        pop[19] = ghostInterface.fGhost.Y_1[idxPopY(txm1, tzm1, 5, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == 0) ? bzm1 : bz))];
        pop[21] = ghostInterface.fGhost.Y_1[idxPopY(txm1, tzp1, 6, ((tx == 0) ? bxm1 : bx)              , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[24] = ghostInterface.fGhost.Y_1[idxPopY(txp1, tzp1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[25] = ghostInterface.fGhost.Y_1[idxPopY(txp1, tzm1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , bym1, ((tz == 0) ? bzm1 : bz))];
        #endif //D3Q27
    }
else if (ty == (BLOCK_NY - 1))
    { //n
        pop[ 4] = ghostInterface.fGhost.Y_0[idxPopY(tx  , tz, 0, bx                                     , byp1, bz)];
        pop[ 8] = ghostInterface.fGhost.Y_0[idxPopY(txp1, tz, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , byp1, bz)];
        pop[12] = ghostInterface.fGhost.Y_0[idxPopY(tx, tzp1, 2, bx                                     , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[13] = ghostInterface.fGhost.Y_0[idxPopY(txm1, tz, 3, ((tx == 0) ? bxm1 : bx)                , byp1, bz)];
        pop[18] = ghostInterface.fGhost.Y_0[idxPopY(tx, tzm1, 4, bx                                     , byp1, ((tz == 0) ? bzm1 : bz))];
        #ifdef D3Q27
        pop[20] = ghostInterface.fGhost.Y_0[idxPopY(txp1, tzp1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        pop[22] = ghostInterface.fGhost.Y_0[idxPopY(txp1, tzm1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , byp1, ((tz == 0) ? bzm1 : bz))];
        pop[23] = ghostInterface.fGhost.Y_0[idxPopY(txm1, tzm1, 7, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == 0) ? bzm1 : bz))];
        pop[26] = ghostInterface.fGhost.Y_0[idxPopY(txm1, tzp1, 8, ((tx == 0) ? bxm1 : bx)              , byp1, ((tz == (BLOCK_NZ - 1)) ? bzp1 : bz))];
        #endif //D3Q27
    }

if (tz == 0)
    { //b
        pop[ 5] = ghostInterface.fGhost.Z_1[idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzm1)];
        pop[ 9] = ghostInterface.fGhost.Z_1[idxPopZ(txm1, ty, 1, ((tx == 0) ? bxm1 : bx)                , by                                    , bzm1)];
        pop[11] = ghostInterface.fGhost.Z_1[idxPopZ(tx, tym1, 2, bx                                     , ((ty == 0) ? bym1 : by)               , bzm1)];
        pop[16] = ghostInterface.fGhost.Z_1[idxPopZ(txp1, ty, 3, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzm1)];
        pop[18] = ghostInterface.fGhost.Z_1[idxPopZ(tx, typ1, 4, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        #ifdef D3Q27
        pop[19] = ghostInterface.fGhost.Z_1[idxPopZ(txm1, tym1, 5, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzm1)];
        pop[22] = ghostInterface.fGhost.Z_1[idxPopZ(txp1, typ1, 6, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        pop[23] = ghostInterface.fGhost.Z_1[idxPopZ(txm1, typ1, 7, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzm1)];
        pop[25] = ghostInterface.fGhost.Z_1[idxPopZ(txp1, tym1, 8, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzm1)];
        #endif //D3Q27
    }
else if (tz == (BLOCK_NZ - 1))
    { //f
        pop[ 6] = ghostInterface.fGhost.Z_0[idxPopZ(tx  , ty, 0, bx                                     , by                                    , bzp1)];
        pop[10] = ghostInterface.fGhost.Z_0[idxPopZ(txp1, ty, 1, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx)   , by                                    , bzp1)];
        pop[12] = ghostInterface.fGhost.Z_0[idxPopZ(tx, typ1, 2, bx                                     , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        pop[15] = ghostInterface.fGhost.Z_0[idxPopZ(txm1, ty, 3, ((tx == 0) ? bxm1 : bx)                , by                                    , bzp1)];
        pop[17] = ghostInterface.fGhost.Z_0[idxPopZ(tx, tym1, 4, bx                                     , ((ty == 0) ? bym1 : by)               , bzp1)];
        #ifdef D3Q27
        pop[20] = ghostInterface.fGhost.Z_0[idxPopZ(txp1, typ1, 5, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        pop[21] = ghostInterface.fGhost.Z_0[idxPopZ(txm1, tym1, 6, ((tx == 0) ? bxm1 : bx)              , ((ty == 0) ? bym1 : by)               , bzp1)];
        pop[24] = ghostInterface.fGhost.Z_0[idxPopZ(txp1, tym1, 7, ((tx == (BLOCK_NX - 1)) ? bxp1 : bx) , ((ty == 0) ? bym1 : by)               , bzp1)];
        pop[26] = ghostInterface.fGhost.Z_0[idxPopZ(txm1, typ1, 8, ((tx == 0) ? bxm1 : bx)              , ((ty == (BLOCK_NY - 1)) ? byp1 : by)  , bzp1)];
        #endif //D3Q27
    }