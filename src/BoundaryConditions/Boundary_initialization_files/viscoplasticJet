// Boundary conditions for the viscoplastic jet case of  Balmforth et al.
// https://doi.org/10.1017/jfm.2016.878 
    
    int hh = (N/2)/2;

    if (z == 0) // B
    {
        nodeType = (BACK);

        if ((y < (NY-1)/2 + hh)&& (y > (NY-1)/2 - hh)){ //slid
        //if((x-(NX-1)/2)*(x-(NX-1)/2) + (y-(NY-1)/2)*(y-(NY-1)/2) < hh*hh){//circular
            nodeType = (BACK  + BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);
        }

    }
    else if (z == (NZ_TOTAL - 1)) // F OUTLET
    {
        nodeType = (FRONT);

        if ((y < (NY-1)/2 + hh)&& (y > (NY-1)/2 - hh)){ //slid
        //if((x-(NX-1)/2)*(x-(NX-1)/2) + (y-(NY-1)/2)*(y-(NY-1)/2) < hh*hh){//circular
            nodeType = (FRONT +  BC_VELOCITY_WALL + BC_VELOCITY_INDEX_0);   
        }
        //nodeType = (FRONT + BC_OUTFLOW);
    }else{        
        nodeType = BULK;   
    }
    