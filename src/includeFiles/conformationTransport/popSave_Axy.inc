if (INTERFACE_BC_WEST) { //w
    ghostInterface.Axy_gGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxyNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Axy_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxyNode[ 8];
    ghostInterface.Axy_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxyNode[10];
    ghostInterface.Axy_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxyNode[14];
    ghostInterface.Axy_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxyNode[16];
    #endif            
}if (INTERFACE_BC_EAST){                    
    ghostInterface.Axy_gGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxyNode[ 1];
    #ifdef D3G19
    ghostInterface.Axy_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxyNode[ 7];
    ghostInterface.Axy_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxyNode[ 9];
    ghostInterface.Axy_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxyNode[13];
    ghostInterface.Axy_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxyNode[15];     
    #endif    
}

if (INTERFACE_BC_SOUTH)  { //s                             
    ghostInterface.Axy_gGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxyNode[ 4];
    #ifdef D3G19
    ghostInterface.Axy_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxyNode[ 8];
    ghostInterface.Axy_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxyNode[12];
    ghostInterface.Axy_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxyNode[13];
    ghostInterface.Axy_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxyNode[18];           
    #endif           
}if (INTERFACE_BC_NORTH){             
    ghostInterface.Axy_gGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxyNode[ 3];
    #ifdef D3G19
    ghostInterface.Axy_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxyNode[ 7];
    ghostInterface.Axy_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxyNode[11];
    ghostInterface.Axy_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxyNode[14];
    ghostInterface.Axy_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxyNode[17];         
    #endif        
}

if (INTERFACE_BC_BACK){ //b                          
    ghostInterface.Axy_gGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxyNode[ 6];
    #ifdef D3G19
    ghostInterface.Axy_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxyNode[10];
    ghostInterface.Axy_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxyNode[12];
    ghostInterface.Axy_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxyNode[15];
    ghostInterface.Axy_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxyNode[17]; 
    #endif    
}if (INTERFACE_BC_FRONT){                  
    ghostInterface.Axy_gGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxyNode[ 5];
    #ifdef D3G19
    ghostInterface.Axy_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxyNode[ 9];
    ghostInterface.Axy_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxyNode[11];
    ghostInterface.Axy_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxyNode[16];
    ghostInterface.Axy_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxyNode[18];    
    #endif                    
}