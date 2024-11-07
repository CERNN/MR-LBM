    /* write to global pop */
    if (INTERFACE_BC_WEST){//w
        velGhostTwo->X_0[idxPopX(ty, tz, 0, bx, by, bz)] = ux_t30;
        velGhostTwo->X_0[idxPopX(ty, tz, 1, bx, by, bz)] = uy_t30;
        velGhostTwo->X_0[idxPopX(ty, tz, 2, bx, by, bz)] = uz_t30;
    }
    if (INTERFACE_BC_EAST){//e
        velGhostTwo->X_1[idxPopX(ty, tz, 0, bx, by, bz)] = ux_t30;
        velGhostTwo->X_1[idxPopX(ty, tz, 1, bx, by, bz)] = uy_t30;
        velGhostTwo->X_1[idxPopX(ty, tz, 2, bx, by, bz)] = uz_t30;
    }

    if (INTERFACE_BC_SOUTH){//s          
        velGhostTwo->Y_0[idxPopY(tx, tz, 0, bx, by, bz)] = ux_t30;
        velGhostTwo->Y_0[idxPopY(tx, tz, 1, bx, by, bz)] = uy_t30;
        velGhostTwo->Y_0[idxPopY(tx, tz, 2, bx, by, bz)] = uz_t30;
    }
    if (INTERFACE_BC_NORTH){//n
        velGhostTwo->Y_1[idxPopY(tx, tz, 0, bx, by, bz)] = ux_t30;
        velGhostTwo->Y_1[idxPopY(tx, tz, 1, bx, by, bz)] = uy_t30;
        velGhostTwo->Y_1[idxPopY(tx, tz, 2, bx, by, bz)] = uz_t30;
    }
    
    if (INTERFACE_BC_BACK){ //b
        velGhostTwo->Z_0[idxPopZ(tx, ty, 0, bx, by, bz)] = ux_t30;
        velGhostTwo->Z_0[idxPopZ(tx, ty, 1, bx, by, bz)] = uy_t30;
        velGhostTwo->Z_0[idxPopZ(tx, ty, 2, bx, by, bz)] = uz_t30;
    }
    if (INTERFACE_BC_FRONT){ //f
        velGhostTwo->Z_1[idxPopZ(tx, ty, 0, bx, by, bz)] = ux_t30;
        velGhostTwo->Z_1[idxPopZ(tx, ty, 1, bx, by, bz)] = uy_t30;
        velGhostTwo->Z_1[idxPopZ(tx, ty, 2, bx, by, bz)] = uz_t30;
    }