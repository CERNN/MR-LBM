    int hh = 16;

    if (z == 0) // B
    {
        nodeType = (BACK);
        
        if((x-(NX-1)/2)*(x-(NX-1)/2) + (y-(NY-1)/2)*(y-(NY-1)/2) < 16*16){
            nodeType = (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);
        }

    }
    else if (z == (NZ_TOTAL - 1)) // F OUTLET
    {
        nodeType = (FRONT);      
    }
    else if (x ==0){
        nodeType = (WEST + BC_OUTFLOW);      
    }
    else if (x ==(NX-1)){
        nodeType = (EAST + BC_OUTFLOW);      
    }
    else if (y ==0){
        nodeType = (SOUTH + BC_OUTFLOW);      
    }
    else if (y ==(NY-1)){
        nodeType = (NORTH + BC_OUTFLOW);      
    }
    else
    {        
        nodeType = BULK;   
    }
    