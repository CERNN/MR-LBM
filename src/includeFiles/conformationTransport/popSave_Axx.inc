if (INTERFACE_BC_WEST) { //w
    ghostInterface.Axx_gGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxxNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Axx_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxxNode[ 8];
    ghostInterface.Axx_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxxNode[10];
    ghostInterface.Axx_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxxNode[14];
    ghostInterface.Axx_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxxNode[16];
    #endif            
}if (INTERFACE_BC_EAST){                    
    ghostInterface.Axx_gGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxxNode[ 1];
    #ifdef D3G19
    ghostInterface.Axx_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxxNode[ 7];
    ghostInterface.Axx_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxxNode[ 9];
    ghostInterface.Axx_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxxNode[13];
    ghostInterface.Axx_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxxNode[15];     
    #endif    
}

if (INTERFACE_BC_SOUTH)  { //s                             
    ghostInterface.Axx_gGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxxNode[ 4];
    #ifdef D3G19
    ghostInterface.Axx_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxxNode[ 8];
    ghostInterface.Axx_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxxNode[12];
    ghostInterface.Axx_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxxNode[13];
    ghostInterface.Axx_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxxNode[18];           
    #endif           
}if (INTERFACE_BC_NORTH){             
    ghostInterface.Axx_gGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxxNode[ 3];
    #ifdef D3G19
    ghostInterface.Axx_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxxNode[ 7];
    ghostInterface.Axx_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxxNode[11];
    ghostInterface.Axx_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxxNode[14];
    ghostInterface.Axx_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxxNode[17];         
    #endif        
}

if (INTERFACE_BC_BACK){ //b                          
    ghostInterface.Axx_gGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxxNode[ 6];
    #ifdef D3G19
    ghostInterface.Axx_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxxNode[10];
    ghostInterface.Axx_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxxNode[12];
    ghostInterface.Axx_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxxNode[15];
    ghostInterface.Axx_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxxNode[17]; 
    #endif    
}if (INTERFACE_BC_FRONT){                  
    ghostInterface.Axx_gGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxxNode[ 5];
    #ifdef D3G19
    ghostInterface.Axx_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxxNode[ 9];
    ghostInterface.Axx_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxxNode[11];
    ghostInterface.Axx_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxxNode[16];
    ghostInterface.Axx_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxxNode[18];    
    #endif                    
}