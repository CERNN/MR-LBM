    /* write to global pop */
    if (INTERFACE_BC_WEST){//w
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2]; 
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
        #ifdef D3Q27
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
        ghostInterface.gGhost.X_0[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27
    }if (INTERFACE_BC_EAST){//e
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
        #ifdef D3Q27
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
        ghostInterface.gGhost.X_1[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27       
    }

    if (INTERFACE_BC_SOUTH){//s          
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 4];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 8];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 2, bx, by, bz)] = pop[12];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 3, bx, by, bz)] = pop[13];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
        ghostInterface.gGhost.Y_0[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27
    }if (INTERFACE_BC_NORTH){//n
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 3];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 7];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 2, bx, by, bz)] = pop[11];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 3, bx, by, bz)] = pop[14];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
        ghostInterface.gGhost.Y_1[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27
    }
    
    if (INTERFACE_BC_BACK){ //b
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 6];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[10];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[12];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[15];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
        ghostInterface.gGhost.Z_0[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27
    }if (INTERFACE_BC_FRONT){
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 5];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[ 9];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[11];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[16];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
        ghostInterface.gGhost.Z_1[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27
    }