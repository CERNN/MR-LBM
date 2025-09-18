int Z_EXPANSION = 4*N;
int H_INLET = (NY-1)/2;

 


if (z == 0) // B
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
  

if(y == (NY-1))
    nodeType = NORTH; //<<<<< OK
if ((y == H_INLET) && (z >= 0) && (z < Z_EXPANSION))
    nodeType = SOUTH; //<<<<< OK
if((y ==0))
    nodeType = SOUTH; //<<<<< OK
if ((z < (Z_EXPANSION)) && (y < H_INLET))
    nodeType = SOLID_NODE; //<<<<< OK
if((z == Z_EXPANSION) && (y == 0))
    nodeType = SOUTH_BACK; //<<<<< OK
if((z == Z_EXPANSION) && (y == H_INLET))
    nodeType = 0b00000011; //<<<<< OK
if((z == Z_EXPANSION) && (y > 0) && (y < H_INLET))
    nodeType = BACK; //<<<<< OK

