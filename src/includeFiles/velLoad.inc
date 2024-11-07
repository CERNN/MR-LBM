
if(tx==0){//w
    sharedVel[idxVelBlock(tx - 1, ty, tz, 0)] = velGhostOne->X_1[idxVelX(ty, tz, 0, bxm1, by, bz)];
    sharedVel[idxVelBlock(tx - 1, ty, tz, 1)] = velGhostOne->X_1[idxVelX(ty, tz, 1, bxm1, by, bz)];
    sharedVel[idxVelBlock(tx - 1, ty, tz, 2)] = velGhostOne->X_1[idxVelX(ty, tz, 2, bxm1, by, bz)];
}else if(tx==(BLOCK_NX-1)){//e
    sharedVel[idxVelBlock(tx + 1, ty, tz, 0)] = velGhostOne->X_0[idxVelX(ty, tz, 0, bxp1, by, bz)];
    sharedVel[idxVelBlock(tx + 1, ty, tz, 1)] = velGhostOne->X_0[idxVelX(ty, tz, 0, bxp1, by, bz)];
    sharedVel[idxVelBlock(tx + 1, ty, tz, 2)] = velGhostOne->X_0[idxVelX(ty, tz, 0, bxp1, by, bz)];
}

if(ty==0){//s
    sharedVel[idxVelBlock(tx, ty - 1, tz, 0)] = velGhostOne->Y_1[idxVelY(tx, tz, 0, bx, bym1, bz)];
    sharedVel[idxVelBlock(tx, ty - 1, tz, 1)] = velGhostOne->Y_1[idxVelY(tx, tz, 0, bx, bym1, bz)];
    sharedVel[idxVelBlock(tx, ty - 1, tz, 2)] = velGhostOne->Y_1[idxVelY(tx, tz, 0, bx, bym1, bz)];
}else if(ty==(BLOCK_NY-1)){//n   
    sharedVel[idxVelBlock(tx, ty + 1, tz, 0)] = velGhostOne->Y_0[idxVelY(tx, tz, 0, bx, byp1, bz)];
    sharedVel[idxVelBlock(tx, ty + 1, tz, 1)] = velGhostOne->Y_0[idxVelY(tx, tz, 0, bx, byp1, bz)];
    sharedVel[idxVelBlock(tx, ty + 1, tz, 2)] = velGhostOne->Y_0[idxVelY(tx, tz, 0, bx, byp1, bz)];
}

if(tz==0){//b
    sharedVel[idxVelBlock(tx, ty, tz - 1, 0)] = velGhostOne->Z_1[idxVelZ(tx, ty, 0, bx, by, bzm1)];
    sharedVel[idxVelBlock(tx, ty, tz - 1, 1)] = velGhostOne->Z_1[idxVelZ(tx, ty, 0, bx, by, bzm1)];
    sharedVel[idxVelBlock(tx, ty, tz - 1, 2)] = velGhostOne->Z_1[idxVelZ(tx, ty, 0, bx, by, bzm1)];
}else if(tz==(BLOCK_NZ-1)){//f
    sharedVel[idxVelBlock(tx, ty, tz + 1, 0)] = velGhostOne->Z_0[idxVelZ(tx, ty, 0, bx, by, bzp1)];
    sharedVel[idxVelBlock(tx, ty, tz + 1, 1)] = velGhostOne->Z_0[idxVelZ(tx, ty, 0, bx, by, bzp1)];
    sharedVel[idxVelBlock(tx, ty, tz + 1, 2)] = velGhostOne->Z_0[idxVelZ(tx, ty, 0, bx, by, bzp1)];
}