if (INTERFACE_BC_WEST) { //w
    ghostInterface.Ayy_gGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AyyNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Ayy_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AyyNode[ 8];
    ghostInterface.Ayy_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AyyNode[10];
    ghostInterface.Ayy_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AyyNode[14];
    ghostInterface.Ayy_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AyyNode[16];
    #endif            
}if (INTERFACE_BC_EAST){                    
    ghostInterface.Ayy_gGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AyyNode[ 1];
    #ifdef D3G19
    ghostInterface.Ayy_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AyyNode[ 7];
    ghostInterface.Ayy_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AyyNode[ 9];
    ghostInterface.Ayy_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AyyNode[13];
    ghostInterface.Ayy_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AyyNode[15];     
    #endif    
}

if (INTERFACE_BC_SOUTH)  { //s                             
    ghostInterface.Ayy_gGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AyyNode[ 4];
    #ifdef D3G19
    ghostInterface.Ayy_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AyyNode[ 8];
    ghostInterface.Ayy_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AyyNode[12];
    ghostInterface.Ayy_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AyyNode[13];
    ghostInterface.Ayy_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AyyNode[18];           
    #endif           
}if (INTERFACE_BC_NORTH){             
    ghostInterface.Ayy_gGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AyyNode[ 3];
    #ifdef D3G19
    ghostInterface.Ayy_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AyyNode[ 7];
    ghostInterface.Ayy_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AyyNode[11];
    ghostInterface.Ayy_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AyyNode[14];
    ghostInterface.Ayy_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AyyNode[17];         
    #endif        
}

if (INTERFACE_BC_BACK){ //b                          
    ghostInterface.Ayy_gGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AyyNode[ 6];
    #ifdef D3G19
    ghostInterface.Ayy_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AyyNode[10];
    ghostInterface.Ayy_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AyyNode[12];
    ghostInterface.Ayy_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AyyNode[15];
    ghostInterface.Ayy_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AyyNode[17]; 
    #endif    
}if (INTERFACE_BC_FRONT){                  
    ghostInterface.Ayy_gGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AyyNode[ 5];
    #ifdef D3G19
    ghostInterface.Ayy_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AyyNode[ 9];
    ghostInterface.Ayy_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AyyNode[11];
    ghostInterface.Ayy_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AyyNode[16];
    ghostInterface.Ayy_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AyyNode[18];    
    #endif                    
}