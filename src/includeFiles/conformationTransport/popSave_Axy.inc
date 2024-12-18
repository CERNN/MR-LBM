if (threadIdx.x == 0) { //w
    ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxyNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxyNode[ 8];
    ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxyNode[10];
    ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxyNode[14];
    ghostInterface.Axy_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxyNode[16];
    #endif            
}else if (threadIdx.x == (BLOCK_NX - 1)){                    
    ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxyNode[ 1];
    #ifdef D3G19
    ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxyNode[ 7];
    ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxyNode[ 9];
    ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxyNode[13];
    ghostInterface.Axy_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxyNode[15];     
    #endif    
}

if (threadIdx.y == 0)  { //s                             
    ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxyNode[ 4];
    #ifdef D3G19
    ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxyNode[ 8];
    ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxyNode[12];
    ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxyNode[13];
    ghostInterface.Axy_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxyNode[18];           
    #endif           
}else if (threadIdx.y == (BLOCK_NY - 1)){             
    ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxyNode[ 3];
    #ifdef D3G19
    ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxyNode[ 7];
    ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxyNode[11];
    ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxyNode[14];
    ghostInterface.Axy_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxyNode[17];         
    #endif        
}

if (threadIdx.z == 0){ //b                          
    ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxyNode[ 6];
    #ifdef D3G19
    ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxyNode[10];
    ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxyNode[12];
    ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxyNode[15];
    ghostInterface.Axy_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxyNode[17]; 
    #endif    
}else if (threadIdx.z == (BLOCK_NZ - 1)){                  
    ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxyNode[ 5];
    #ifdef D3G19
    ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxyNode[ 9];
    ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxyNode[11];
    ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxyNode[16];
    ghostInterface.Axy_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxyNode[18];    
    #endif                    
}