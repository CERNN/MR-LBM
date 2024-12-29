// TODO NOT WORKING. NEED PROPER TREATMENT OF THE GRADIENT OF THE CONFORMATION TENSOR
// THE PROBLEM RESIDES IN HOW TO OBTAIN THE VALUES OF THE NEIGHBORING LATTICES WITHOUT COMPUTINGT THE EIGIN VALUES AND EIGEN VECTORS AGAIN.

s_pop[idxConfBlock(tx, ty, tz, 0)] = Cxx;
s_pop[idxConfBlock(tx, ty, tz, 1)] = Cxy;
s_pop[idxConfBlock(tx, ty, tz, 2)] = Cxz;
s_pop[idxConfBlock(tx, ty, tz, 3)] = Cyy;
s_pop[idxConfBlock(tx, ty, tz, 4)] = Cyz;
s_pop[idxConfBlock(tx, ty, tz, 5)] = Czz;

// Load neighboring halo cells in shared memory
#include "includeFiles/conformationTransport\confLoad.inc"

__syncthreads();

// Compute unrolled gradient calculations
//FINITE CENTRAL DIFFERENCE ORDER 1, otherwise i would have to store a halo with width of 2 in shared memory
dfloat dCxxdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 0)] - s_pop[idxConfBlock(tx - 1, ty, tz, 0)])/2;
dfloat dCxydx = (s_pop[idxConfBlock(tx + 1, ty, tz, 1)] - s_pop[idxConfBlock(tx - 1, ty, tz, 1)])/2;
dfloat dCxzdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 2)] - s_pop[idxConfBlock(tx - 1, ty, tz, 2)])/2;

dfloat dCxydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 1)] - s_pop[idxConfBlock(tx, ty - 1, tz, 1)])/2;
dfloat dCyydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 3)] - s_pop[idxConfBlock(tx, ty - 1, tz, 3)])/2;
dfloat dCyzdy = (s_pop[idxConfBlock(tx, ty + 1, tz, 4)] - s_pop[idxConfBlock(tx, ty - 1, tz, 4)])/2;

dfloat dCxzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 2)] - s_pop[idxConfBlock(tx, ty, tz - 1, 2)])/2;
dfloat dCyzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 4)] - s_pop[idxConfBlock(tx, ty, tz - 1, 4)])/2;
dfloat dCzzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 5)] - s_pop[idxConfBlock(tx, ty, tz - 1, 5)])/2;

//if(x == 32 && z == 32)
//    printf("y: %d dAyydy %f Ayy_p1 %f Ayy %f Ayy_m1 %f  \n",y,dAyydy,s_pop[idxConfBlock(tx, ty + 1, tz, 3)],s_pop[idxConfBlock(tx, ty, tz, 3)],s_pop[idxConfBlock(tx, ty - 1, tz, 3)]);
//
//OVERWRITE FOR NEAR WALL NODES
if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dCxxdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 0)] - s_pop[idxConfBlock(tx, ty, tz, 0)]);
    dCxydx = (s_pop[idxConfBlock(tx + 1, ty, tz, 1)] - s_pop[idxConfBlock(tx, ty, tz, 1)]);
    dCxzdx = (s_pop[idxConfBlock(tx + 1, ty, tz, 2)] - s_pop[idxConfBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dCxxdx = (s_pop[idxConfBlock(tx, ty, tz, 0)] - s_pop[idxConfBlock(tx - 1, ty, tz, 0)]);
    dCxydx = (s_pop[idxConfBlock(tx, ty, tz, 1)] - s_pop[idxConfBlock(tx - 1, ty, tz, 1)]);
    dCxzdx = (s_pop[idxConfBlock(tx, ty, tz, 2)] - s_pop[idxConfBlock(tx - 1, ty, tz, 2)]);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dCxydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 1)] - s_pop[idxConfBlock(tx, ty, tz, 1)]);
    dCyydy = (s_pop[idxConfBlock(tx, ty + 1, tz, 3)] - s_pop[idxConfBlock(tx, ty, tz, 3)]);
    dCyzdy = (s_pop[idxConfBlock(tx, ty + 1, tz, 4)] - s_pop[idxConfBlock(tx, ty, tz, 4)]);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dCxydy = (s_pop[idxConfBlock(tx, ty, tz, 1)] - s_pop[idxConfBlock(tx, ty - 1, tz, 1)]);
    dCyydy = (s_pop[idxConfBlock(tx, ty, tz, 3)] - s_pop[idxConfBlock(tx, ty - 1, tz, 3)]);
    dCyzdy = (s_pop[idxConfBlock(tx, ty, tz, 4)] - s_pop[idxConfBlock(tx, ty - 1, tz, 4)]);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dCxzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 2)] - s_pop[idxConfBlock(tx, ty, tz, 2)]);
    dCyzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 4)] - s_pop[idxConfBlock(tx, ty, tz, 4)]);
    dCzzdz = (s_pop[idxConfBlock(tx, ty, tz + 1, 5)] - s_pop[idxConfBlock(tx, ty, tz, 5)]);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dCxzdz = (s_pop[idxConfBlock(tx, ty, tz, 2)] - s_pop[idxConfBlock(tx, ty, tz - 1, 2)]);
    dCyzdz = (s_pop[idxConfBlock(tx, ty, tz, 4)] - s_pop[idxConfBlock(tx, ty, tz - 1, 4)]);
    dCzzdz = (s_pop[idxConfBlock(tx, ty, tz, 5)] - s_pop[idxConfBlock(tx, ty, tz - 1, 5)]);
}


//if(x == 4 && y == 4 && z == 4 )
//    printf("step %d dAxxdx %f \n",step,dAxxdx);