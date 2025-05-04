    /* write to global pop */
    if (INTERFACE_BC_WEST){//w
        ghostInterface.conf_gGhost.X_0[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
        ghostInterface.conf_gGhost.X_0[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
        ghostInterface.conf_gGhost.X_0[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
        ghostInterface.conf_gGhost.X_0[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
        ghostInterface.conf_gGhost.X_0[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
        ghostInterface.conf_gGhost.X_0[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
    }
    if (INTERFACE_BC_EAST){//e
        ghostInterface.conf_gGhost.X_1[g_idxConfX(ty, tz, 0, bx, by, bz)] = AxxVar;
        ghostInterface.conf_gGhost.X_1[g_idxConfX(ty, tz, 1, bx, by, bz)] = AxyVar;
        ghostInterface.conf_gGhost.X_1[g_idxConfX(ty, tz, 2, bx, by, bz)] = AxzVar;
        ghostInterface.conf_gGhost.X_1[g_idxConfX(ty, tz, 3, bx, by, bz)] = AyyVar;
        ghostInterface.conf_gGhost.X_1[g_idxConfX(ty, tz, 4, bx, by, bz)] = AyzVar;
        ghostInterface.conf_gGhost.X_1[g_idxConfX(ty, tz, 5, bx, by, bz)] = AzzVar;
    }

    if (INTERFACE_BC_SOUTH){//s          
        ghostInterface.conf_gGhost.Y_0[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
        ghostInterface.conf_gGhost.Y_0[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
        ghostInterface.conf_gGhost.Y_0[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
        ghostInterface.conf_gGhost.Y_0[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
        ghostInterface.conf_gGhost.Y_0[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
        ghostInterface.conf_gGhost.Y_0[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
    }
    if (INTERFACE_BC_NORTH){//n
        ghostInterface.conf_gGhost.Y_1[g_idxConfY(tx, tz, 0, bx, by, bz)] = AxxVar;
        ghostInterface.conf_gGhost.Y_1[g_idxConfY(tx, tz, 1, bx, by, bz)] = AxyVar;
        ghostInterface.conf_gGhost.Y_1[g_idxConfY(tx, tz, 2, bx, by, bz)] = AxzVar;
        ghostInterface.conf_gGhost.Y_1[g_idxConfY(tx, tz, 3, bx, by, bz)] = AyyVar;
        ghostInterface.conf_gGhost.Y_1[g_idxConfY(tx, tz, 4, bx, by, bz)] = AyzVar;
        ghostInterface.conf_gGhost.Y_1[g_idxConfY(tx, tz, 5, bx, by, bz)] = AzzVar;
    }
    
    if (INTERFACE_BC_BACK){ //b
        ghostInterface.conf_gGhost.Z_0[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
        ghostInterface.conf_gGhost.Z_0[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
        ghostInterface.conf_gGhost.Z_0[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
        ghostInterface.conf_gGhost.Z_0[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
        ghostInterface.conf_gGhost.Z_0[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
        ghostInterface.conf_gGhost.Z_0[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
    }
    if (INTERFACE_BC_FRONT){ //f
        ghostInterface.conf_gGhost.Z_1[g_idxConfZ(tx, ty, 0, bx, by, bz)] = AxxVar;
        ghostInterface.conf_gGhost.Z_1[g_idxConfZ(tx, ty, 1, bx, by, bz)] = AxyVar;
        ghostInterface.conf_gGhost.Z_1[g_idxConfZ(tx, ty, 2, bx, by, bz)] = AxzVar;
        ghostInterface.conf_gGhost.Z_1[g_idxConfZ(tx, ty, 3, bx, by, bz)] = AyyVar;
        ghostInterface.conf_gGhost.Z_1[g_idxConfZ(tx, ty, 4, bx, by, bz)] = AyzVar;
        ghostInterface.conf_gGhost.Z_1[g_idxConfZ(tx, ty, 5, bx, by, bz)] = AzzVar;
    }