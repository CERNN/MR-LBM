if (threadIdx.x == 0) { //w
    ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AzzNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AzzNode[ 8];
    ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AzzNode[10];
    ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AzzNode[14];
    ghostInterface.Azz_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AzzNode[16];
    #endif            
}else if (threadIdx.x == (BLOCK_NX - 1)){                    
    ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AzzNode[ 1];
    #ifdef D3G19
    ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AzzNode[ 7];
    ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AzzNode[ 9];
    ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AzzNode[13];
    ghostInterface.Azz_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AzzNode[15];     
    #endif    
}

if (threadIdx.y == 0)  { //s                             
    ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AzzNode[ 4];
    #ifdef D3G19
    ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AzzNode[ 8];
    ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AzzNode[12];
    ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AzzNode[13];
    ghostInterface.Azz_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AzzNode[18];           
    #endif           
}else if (threadIdx.y == (BLOCK_NY - 1)){             
    ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AzzNode[ 3];
    #ifdef D3G19
    ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AzzNode[ 7];
    ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AzzNode[11];
    ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AzzNode[14];
    ghostInterface.Azz_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AzzNode[17];         
    #endif        
}

if (threadIdx.z == 0){ //b                          
    ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AzzNode[ 6];
    #ifdef D3G19
    ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AzzNode[10];
    ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AzzNode[12];
    ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AzzNode[15];
    ghostInterface.Azz_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AzzNode[17]; 
    #endif    
}else if (threadIdx.z == (BLOCK_NZ - 1)){                  
    ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AzzNode[ 5];
    #ifdef D3G19
    ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AzzNode[ 9];
    ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AzzNode[11];
    ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AzzNode[16];
    ghostInterface.Azz_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AzzNode[18];    
    #endif                    
}