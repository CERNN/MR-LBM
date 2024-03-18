    if (y == 0) // F OUTLET
    {
        nodeType = (SOUTH + BC_VELOCITY_WALL);      

        if ((x>(95))&&(x<(160))&&(z>(95))&&(z<(160))){
                nodeType = (SOLID_NODE);
        }
    }else if (y == (NY - 1)) // F OUTLET
    {
        nodeType = (NORTH + BC_OUTFLOW);      
    }else if  (z == 0) // B
    {
        nodeType = (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);
    }
    else if (z == (NZ_TOTAL - 1)) // F OUTLET
    {
        nodeType = (FRONT + BC_OUTFLOW);      
    }
    else
    {        
        nodeType = BULK;   
    }