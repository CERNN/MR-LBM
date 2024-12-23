if (INTERFACE_BC_WEST) { //w
    ghostInterface.Axz_gGhost.X_0[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxzNode[ 2]; 
    #ifdef D3G19
    ghostInterface.Axz_gGhost.X_0[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxzNode[ 8];
    ghostInterface.Axz_gGhost.X_0[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxzNode[10];
    ghostInterface.Axz_gGhost.X_0[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxzNode[14];
    ghostInterface.Axz_gGhost.X_0[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxzNode[16];
    #endif            
}if (INTERFACE_BC_EAST){                    
    ghostInterface.Axz_gGhost.X_1[g_idxPopX(ty, tz, 0, bx, by, bz)] = AxzNode[ 1];
    #ifdef D3G19
    ghostInterface.Axz_gGhost.X_1[g_idxPopX(ty, tz, 1, bx, by, bz)] = AxzNode[ 7];
    ghostInterface.Axz_gGhost.X_1[g_idxPopX(ty, tz, 2, bx, by, bz)] = AxzNode[ 9];
    ghostInterface.Axz_gGhost.X_1[g_idxPopX(ty, tz, 3, bx, by, bz)] = AxzNode[13];
    ghostInterface.Axz_gGhost.X_1[g_idxPopX(ty, tz, 4, bx, by, bz)] = AxzNode[15];     
    #endif    
}

if (INTERFACE_BC_SOUTH)  { //s                             
    ghostInterface.Axz_gGhost.Y_0[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxzNode[ 4];
    #ifdef D3G19
    ghostInterface.Axz_gGhost.Y_0[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxzNode[ 8];
    ghostInterface.Axz_gGhost.Y_0[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxzNode[12];
    ghostInterface.Axz_gGhost.Y_0[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxzNode[13];
    ghostInterface.Axz_gGhost.Y_0[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxzNode[18];           
    #endif           
}if (INTERFACE_BC_NORTH){             
    ghostInterface.Axz_gGhost.Y_1[g_idxPopY(tx, tz, 0, bx, by, bz)] = AxzNode[ 3];
    #ifdef D3G19
    ghostInterface.Axz_gGhost.Y_1[g_idxPopY(tx, tz, 1, bx, by, bz)] = AxzNode[ 7];
    ghostInterface.Axz_gGhost.Y_1[g_idxPopY(tx, tz, 2, bx, by, bz)] = AxzNode[11];
    ghostInterface.Axz_gGhost.Y_1[g_idxPopY(tx, tz, 3, bx, by, bz)] = AxzNode[14];
    ghostInterface.Axz_gGhost.Y_1[g_idxPopY(tx, tz, 4, bx, by, bz)] = AxzNode[17];         
    #endif        
}

if (INTERFACE_BC_BACK){ //b                          
    ghostInterface.Axz_gGhost.Z_0[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxzNode[ 6];
    #ifdef D3G19
    ghostInterface.Axz_gGhost.Z_0[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxzNode[10];
    ghostInterface.Axz_gGhost.Z_0[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxzNode[12];
    ghostInterface.Axz_gGhost.Z_0[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxzNode[15];
    ghostInterface.Axz_gGhost.Z_0[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxzNode[17]; 
    #endif    
}if (INTERFACE_BC_FRONT){                  
    ghostInterface.Axz_gGhost.Z_1[g_idxPopZ(tx, ty, 0, bx, by, bz)] = AxzNode[ 5];
    #ifdef D3G19
    ghostInterface.Axz_gGhost.Z_1[g_idxPopZ(tx, ty, 1, bx, by, bz)] = AxzNode[ 9];
    ghostInterface.Axz_gGhost.Z_1[g_idxPopZ(tx, ty, 2, bx, by, bz)] = AxzNode[11];
    ghostInterface.Axz_gGhost.Z_1[g_idxPopZ(tx, ty, 3, bx, by, bz)] = AxzNode[16];
    ghostInterface.Axz_gGhost.Z_1[g_idxPopZ(tx, ty, 4, bx, by, bz)] = AxzNode[18];    
    #endif                    
}