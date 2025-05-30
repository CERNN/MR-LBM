    if (y == 0 && x == 0 && z == 0) // SWB
    {
        nodeType = SOUTH_WEST_BACK;
    }
    else if (y == 0 && x == 0 && z == (NZ_TOTAL - 1)) // SWF
    {
        nodeType = SOUTH_WEST_FRONT;
    }
    else if (y == 0 && x == (NX - 1) && z == 0) // SEB
    {
        nodeType = SOUTH_EAST_BACK;
    }
    else if (y == 0 && x == (NX - 1) && z == (NZ_TOTAL - 1)) // SEF
    {
        nodeType = SOUTH_EAST_FRONT;
    }
    else if (y == (NY - 1) && x == 0 && z == 0) // NWB
    {
        nodeType = NORTH;
    }
    else if (y == (NY - 1) && x == 0 && z == (NZ_TOTAL - 1)) // NWF
    {
        nodeType = NORTH;
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == 0) // NEB
    {
        nodeType = NORTH;
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == (NZ_TOTAL - 1)) // NEF
    {
        nodeType = NORTH;
    }
    else if (y == 0 && x == 0) // SW
    {
        nodeType = SOUTH_WEST;
    }
    else if (y == 0 && x == (NX - 1)) // SE
    {
        nodeType = SOUTH_EAST;
    }
    else if (y == (NY - 1) && x == 0) // NW
    {
         nodeType = NORTH;
    }
    else if (y == (NY - 1) && x == (NX - 1)) // NE
    {
        nodeType = NORTH;
    }
    else if (y == 0 && z == 0) // SB
    {
        nodeType = SOUTH_BACK;
    }
    else if (y == 0 && z == (NZ_TOTAL - 1)) // SF
    {
        nodeType = SOUTH_FRONT;
    }
    else if (y == (NY - 1) && z == 0) // NB
    {
        nodeType = NORTH;
    }
    else if (y == (NY - 1) && z == (NZ_TOTAL - 1)) // NF
    {
        nodeType = NORTH;
    }
    else if (x == 0 && z == 0) // WB
    {
        nodeType = WEST_BACK;
    }
    else if (x == 0 && z == (NZ_TOTAL - 1)) // WF
    {
        nodeType = WEST_FRONT;
    }
    else if (x == (NX - 1) && z == 0) // EB
    {
        nodeType = EAST_BACK;
    }
    else if (x == (NX - 1) && z == (NZ_TOTAL - 1)) // EF
    {
        nodeType = EAST_FRONT;
    }
    else if (y == 0) // S
    {
        nodeType = SOUTH;
    }
    else if (y == (NY - 1)) // N
    {
        nodeType = NORTH;
    }
    else if (x == 0) // W
    {
        nodeType = WEST;
    }
    else if (x == (NX - 1)) // E
    {
        nodeType = EAST;
    }
    else if (z == 0) // B
    {
        nodeType = BACK;
    }
    else if (z == (NZ_TOTAL - 1)) // F
    {
        nodeType = FRONT;      
    }
    else{
        nodeType = BULK;
    }
