    //Jet radius of 16
    // not validated yet
    
    
    if (z == 0) // B
    {
        nodeType = (BACK);
        
        if((x-(NX-1)/2)*(x-(NX-1)/2) + (y-(NY-1)/2)*(y-(NY-1)/2) < 16*16){
            nodeType = (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);
        }

    }
    else if (z == (NZ_TOTAL - 1)) // F OUTLET
    {
        nodeType = (FRONT + BC_OUTFLOW);      
    }
    else
    {        
        nodeType = BULK;   
    }
    