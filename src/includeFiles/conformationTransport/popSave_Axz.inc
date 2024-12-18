if (threadIdx.x == 0) { //w
    ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxzNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxzNode[ 8];
    ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxzNode[10];
    ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxzNode[14];
    ghostInterface.Axz_fGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxzNode[16];
    #endif            
}else if (threadIdx.x == (BLOCK_NX - 1)){                    
    ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxzNode[ 1];
    #ifdef D3G19
    ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxzNode[ 7];
    ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxzNode[ 9];
    ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxzNode[13];
    ghostInterface.Axz_fGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxzNode[15];     
    #endif    
}

if (threadIdx.y == 0)  { //s                             
    ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxzNode[ 4];
    #ifdef D3G19
    ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxzNode[ 8];
    ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxzNode[12];
    ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxzNode[13];
    ghostInterface.Axz_fGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxzNode[18];           
    #endif           
}else if (threadIdx.y == (BLOCK_NY - 1)){             
    ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxzNode[ 3];
    #ifdef D3G19
    ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxzNode[ 7];
    ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxzNode[11];
    ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxzNode[14];
    ghostInterface.Axz_fGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxzNode[17];         
    #endif        
}

if (threadIdx.z == 0){ //b                          
    ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxzNode[ 6];
    #ifdef D3G19
    ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxzNode[10];
    ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxzNode[12];
    ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxzNode[15];
    ghostInterface.Axz_fGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxzNode[17]; 
    #endif    
}else if (threadIdx.z == (BLOCK_NZ - 1)){                  
    ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxzNode[ 5];
    #ifdef D3G19
    ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxzNode[ 9];
    ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxzNode[11];
    ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxzNode[16];
    ghostInterface.Axz_fGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxzNode[18];    
    #endif                    
}