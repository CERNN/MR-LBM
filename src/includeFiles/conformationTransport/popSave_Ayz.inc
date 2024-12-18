if (threadIdx.x == 0) { //w
    ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AyzNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AyzNode[ 8];
    ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AyzNode[10];
    ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AyzNode[14];
    ghostInterface.Ayz_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AyzNode[16];
    #endif            
}else if (threadIdx.x == (BLOCK_NX - 1)){                    
    ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AyzNode[ 1];
    #ifdef D3G19
    ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AyzNode[ 7];
    ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AyzNode[ 9];
    ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AyzNode[13];
    ghostInterface.Ayz_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AyzNode[15];     
    #endif    
}

if (threadIdx.y == 0)  { //s                             
    ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AyzNode[ 4];
    #ifdef D3G19
    ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AyzNode[ 8];
    ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AyzNode[12];
    ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AyzNode[13];
    ghostInterface.Ayz_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AyzNode[18];           
    #endif           
}else if (threadIdx.y == (BLOCK_NY - 1)){             
    ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AyzNode[ 3];
    #ifdef D3G19
    ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AyzNode[ 7];
    ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AyzNode[11];
    ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AyzNode[14];
    ghostInterface.Ayz_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AyzNode[17];         
    #endif        
}

if (threadIdx.z == 0){ //b                          
    ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AyzNode[ 6];
    #ifdef D3G19
    ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AyzNode[10];
    ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AyzNode[12];
    ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AyzNode[15];
    ghostInterface.Ayz_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AyzNode[17]; 
    #endif    
}else if (threadIdx.z == (BLOCK_NZ - 1)){                  
    ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AyzNode[ 5];
    #ifdef D3G19
    ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AyzNode[ 9];
    ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AyzNode[11];
    ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AyzNode[16];
    ghostInterface.Ayz_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AyzNode[18];    
    #endif                    
}