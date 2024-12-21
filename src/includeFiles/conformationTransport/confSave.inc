if (threadIdx.x == 0) { //w
    ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
    ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
    ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
    ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
    ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
    ghostInterface.conf_fGhost.X_0[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
}else if (threadIdx.x == (BLOCK_NX - 1)){                    
    ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
    ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
    ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
    ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
    ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
    ghostInterface.conf_fGhost.X_1[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
}

if (threadIdx.y == 0)  { //s                             
    ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
    ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
    ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
    ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
    ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
    ghostInterface.conf_fGhost.Y_0[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
}else if (threadIdx.y == (BLOCK_NY - 1)){             
    ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
    ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
    ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
    ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
    ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
    ghostInterface.conf_fGhost.Y_1[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
}

if (threadIdx.z == 0){ //b                          
    ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
    ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
    ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
    ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
    ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
    ghostInterface.conf_fGhost.Z_0[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
}else if (threadIdx.z == (BLOCK_NZ - 1)){                  
    ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
    ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
    ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
    ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
    ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
    ghostInterface.conf_fGhost.Z_1[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
}