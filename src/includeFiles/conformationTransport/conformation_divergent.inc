s_pop[idxConfBlock(tx, ty, tz, 0)] = AxxVar;
s_pop[idxConfBlock(tx, ty, tz, 1)] = AxyVar;
s_pop[idxConfBlock(tx, ty, tz, 2)] = AxzVar;
s_pop[idxConfBlock(tx, ty, tz, 3)] = AyyVar;
s_pop[idxConfBlock(tx, ty, tz, 4)] = AyzVar;
s_pop[idxConfBlock(tx, ty, tz, 5)] = AzzVar;

// Load neighboring halo cells in shared memory
#include "includeFiles/conformationTransport\confLoad.inc"

__syncthreads();

// Compute unrolled divergent calculations
//FINITE CENTRAL DIFFERENCE ORDER 1, otherwise i would have to store a halo with width of 2 in shared memory
dfloat dAxxdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 0)] - s_pop[idxConfBlock(tx - 1, ty, tz, 0)])/2;
dfloat dAxydx = (s_pop[idxConfBlock(tx + 1, ty, tz, 1)] - s_pop[idxConfBlock(tx - 1, ty, tz, 1)])/2;
dfloat dAxzdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 2)] - s_pop[idxConfBlock(tx - 1, ty, tz, 2)])/2;

dfloat dAxydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 1)] - s_pop[idxConfBlock(tx, ty - 1, tz, 1)])/2;
dfloat dAyydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 3)] - s_pop[idxConfBlock(tx, ty - 1, tz, 3)])/2;
dfloat dAyzdy = (s_pop[idxConfBlock(tx, ty + 1, tz, 4)] - s_pop[idxConfBlock(tx, ty - 1, tz, 4)])/2;

dfloat dAxzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 2)] - s_pop[idxConfBlock(tx, ty, tz - 1, 2)])/2;
dfloat dAyzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 4)] - s_pop[idxConfBlock(tx, ty, tz - 1, 4)])/2;
dfloat dAzzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 5)] - s_pop[idxConfBlock(tx, ty, tz - 1, 5)])/2;

//if(x == 32 && z == 32)
//    printf("y: %d dAyydy %f Ayy_p1 %f Ayy %f Ayy_m1 %f  \n",y,dAyydy,s_pop[idxConfBlock(tx, ty + 1, tz, 3)],s_pop[idxConfBlock(tx, ty, tz, 3)],s_pop[idxConfBlock(tx, ty - 1, tz, 3)]);
//
//OVERWRITE FOR NEAR WALL NODES
if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dAxxdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 0)] - s_pop[idxConfBlock(tx, ty, tz, 0)]);
    dAxydx = (s_pop[idxConfBlock(tx + 1, ty, tz, 1)] - s_pop[idxConfBlock(tx, ty, tz, 1)]);
    dAxzdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 2)] - s_pop[idxConfBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dAxxdx = (s_pop[idxConfBlock(tx, ty, tz, 0)] - s_pop[idxConfBlock(tx - 1, ty, tz, 0)]);
    dAxydx = (s_pop[idxConfBlock(tx, ty, tz, 1)] - s_pop[idxConfBlock(tx - 1, ty, tz, 1)]);
    dAxzdx = (s_pop[idxConfBlock(tx, ty, tz, 2)] - s_pop[idxConfBlock(tx - 1, ty, tz, 2)]);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dAxydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 1)] - s_pop[idxConfBlock(tx, ty, tz, 1)]);
    dAyydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 3)] - s_pop[idxConfBlock(tx, ty, tz, 3)]);
    dAyzdy = (s_pop[idxConfBlock(tx, ty + 1, tz, 4)] - s_pop[idxConfBlock(tx, ty, tz, 4)]);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dAxydy = (s_pop[idxConfBlock(tx, ty, tz, 1)] - s_pop[idxConfBlock(tx, ty - 1, tz, 1)]);
    dAyydy = (s_pop[idxConfBlock(tx, ty, tz, 3)] - s_pop[idxConfBlock(tx, ty - 1, tz, 3)]);
    dAyzdy = (s_pop[idxConfBlock(tx, ty, tz, 4)] - s_pop[idxConfBlock(tx, ty - 1, tz, 4)]);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dAxzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 2)] - s_pop[idxConfBlock(tx, ty, tz, 2)]);
    dAyzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 4)] - s_pop[idxConfBlock(tx, ty, tz, 4)]);
    dAzzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 5)] - s_pop[idxConfBlock(tx, ty, tz, 5)]);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dAxzdz = (s_pop[idxConfBlock(tx, ty, tz, 2)] - s_pop[idxConfBlock(tx, ty, tz - 1, 2)]);
    dAyzdz = (s_pop[idxConfBlock(tx, ty, tz, 4)] - s_pop[idxConfBlock(tx, ty, tz - 1, 4)]);
    dAzzdz = (s_pop[idxConfBlock(tx, ty, tz, 5)] - s_pop[idxConfBlock(tx, ty, tz - 1, 5)]);
}


//if(x == 4 && y == 4 && z == 4 )
//    printf("step %d dAxxdx %f \n",step,dAxxdx);