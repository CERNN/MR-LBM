    /* write to global pop */
    if (INTERFACE_BC_WEST){//w
        ghostInterface.g_uGhost.X_0[g_idxUX(ty, tz, 0, bx, by, bz)] = ux_t30;
        ghostInterface.g_uGhost.X_0[g_idxUX(ty, tz, 1, bx, by, bz)] = uy_t30;
        ghostInterface.g_uGhost.X_0[g_idxUX(ty, tz, 2, bx, by, bz)] = uz_t30;
    }
    if (INTERFACE_BC_EAST){//e
        ghostInterface.g_uGhost.X_1[g_idxUX(ty, tz, 0, bx, by, bz)] = ux_t30;
        ghostInterface.g_uGhost.X_1[g_idxUX(ty, tz, 1, bx, by, bz)] = uy_t30;
        ghostInterface.g_uGhost.X_1[g_idxUX(ty, tz, 2, bx, by, bz)] = uz_t30;
    }

    if (INTERFACE_BC_SOUTH){//s          
        ghostInterface.g_uGhost.Y_0[g_idxUY(tx, tz, 0, bx, by, bz)] = ux_t30;
        ghostInterface.g_uGhost.Y_0[g_idxUY(tx, tz, 1, bx, by, bz)] = uy_t30;
        ghostInterface.g_uGhost.Y_0[g_idxUY(tx, tz, 2, bx, by, bz)] = uz_t30;
    }
    if (INTERFACE_BC_NORTH){//n
        ghostInterface.g_uGhost.Y_1[g_idxUY(tx, tz, 0, bx, by, bz)] = ux_t30;
        ghostInterface.g_uGhost.Y_1[g_idxUY(tx, tz, 1, bx, by, bz)] = uy_t30;
        ghostInterface.g_uGhost.Y_1[g_idxUY(tx, tz, 2, bx, by, bz)] = uz_t30;
    }
    
    if (INTERFACE_BC_BACK){ //b
        ghostInterface.g_uGhost.Z_0[g_idxUZ(tx, ty, 0, bx, by, bz)] = ux_t30;
        ghostInterface.g_uGhost.Z_0[g_idxUZ(tx, ty, 1, bx, by, bz)] = uy_t30;
        ghostInterface.g_uGhost.Z_0[g_idxUZ(tx, ty, 2, bx, by, bz)] = uz_t30;
    }
    if (INTERFACE_BC_FRONT){ //f
        ghostInterface.g_uGhost.Z_1[g_idxUZ(tx, ty, 0, bx, by, bz)] = ux_t30;
        ghostInterface.g_uGhost.Z_1[g_idxUZ(tx, ty, 1, bx, by, bz)] = uy_t30;
        ghostInterface.g_uGhost.Z_1[g_idxUZ(tx, ty, 2, bx, by, bz)] = uz_t30;
    }