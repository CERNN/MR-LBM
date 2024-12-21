//define shared memory
__shared__ dfloat sharedConf[CONFORMATION_GRAD_BLOCK_SIZE];

sharedConf[idxConfBlock(tx, ty, tz, 0)] = AxxVar;
sharedConf[idxConfBlock(tx, ty, tz, 1)] = AxyVar;
sharedConf[idxConfBlock(tx, ty, tz, 2)] = AxzVar;
sharedConf[idxConfBlock(tx, ty, tz, 3)] = AyyVar;
sharedConf[idxConfBlock(tx, ty, tz, 4)] = AyzVar;
sharedConf[idxConfBlock(tx, ty, tz, 5)] = AzzVar;

// Load neighboring halo cells in shared memory
#include "includeFiles\conformationTransport\confLoad.inc"

__syncthreads();

// Compute unrolled gradient calculations
//FINITE CENTRAL DIFFERENCE ORDER 1, otherwise i would have to store a halo with width of 2 in shared memory
dfloat dAxxdx = (sharedConf[idxConfBlock(tx + 1, ty, tz, 0)] - sharedConf[idxConfBlock(tx - 1, ty, tz, 0)])/2;
dfloat dAxydx = (sharedConf[idxConfBlock(tx + 1, ty, tz, 1)] - sharedConf[idxConfBlock(tx - 1, ty, tz, 1)])/2;
dfloat dAxzdx = (sharedConf[idxConfBlock(tx + 1, ty, tz, 2)] - sharedConf[idxConfBlock(tx - 1, ty, tz, 2)])/2;

dfloat dAxydy = (sharedConf[idxConfBlock(tx, ty + 1, tz, 1)] - sharedConf[idxConfBlock(tx, ty - 1, tz, 1)])/2;
dfloat dAyydy = (sharedConf[idxConfBlock(tx, ty + 1, tz, 3)] - sharedConf[idxConfBlock(tx, ty - 1, tz, 3)])/2;
dfloat dAyzdy = (sharedConf[idxConfBlock(tx, ty + 1, tz, 4)] - sharedConf[idxConfBlock(tx, ty - 1, tz, 4)])/2;

dfloat dAxzdz = (sharedConf[idxConfBlock(tx, ty, tz + 1, 2)] - sharedConf[idxConfBlock(tx, ty, tz - 1, 2)])/2;
dfloat dAyzdz = (sharedConf[idxConfBlock(tx, ty, tz + 1, 4)] - sharedConf[idxConfBlock(tx, ty, tz - 1, 4)])/2;
dfloat dAzzdz = (sharedConf[idxConfBlock(tx, ty, tz + 1, 5)] - sharedConf[idxConfBlock(tx, ty, tz - 1, 5)])/2;

//OVERWRITE FOR NEAR WALL NODES
if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    dAxxdx = (sharedConf[idxConfBlock(tx + 1, ty, tz, 0)] - sharedConf[idxConfBlock(tx, ty, tz, 0)]);
    dAxydx = (sharedConf[idxConfBlock(tx + 1, ty, tz, 1)] - sharedConf[idxConfBlock(tx, ty, tz, 1)]);
    dAxzdx = (sharedConf[idxConfBlock(tx + 1, ty, tz, 2)] - sharedConf[idxConfBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    dAxxdx = (sharedConf[idxConfBlock(tx, ty, tz, 0)] - sharedConf[idxConfBlock(tx - 1, ty, tz, 0)]);
    dAxydx = (sharedConf[idxConfBlock(tx, ty, tz, 1)] - sharedConf[idxConfBlock(tx - 1, ty, tz, 1)]);
    dAxzdx = (sharedConf[idxConfBlock(tx, ty, tz, 2)] - sharedConf[idxConfBlock(tx - 1, ty, tz, 2)]);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    dAxydy = (sharedConf[idxConfBlock(tx, ty + 1, tz, 1)] - sharedConf[idxConfBlock(tx, ty, tz, 1)]);
    dAyydy = (sharedConf[idxConfBlock(tx, ty + 1, tz, 3)] - sharedConf[idxConfBlock(tx, ty, tz, 3)]);
    dAyzdy = (sharedConf[idxConfBlock(tx, ty + 1, tz, 4)] - sharedConf[idxConfBlock(tx, ty, tz, 4)]);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    dAxydy = (sharedConf[idxConfBlock(tx, ty, tz, 1)] - sharedConf[idxConfBlock(tx, ty - 1, tz, 1)]);
    dAyydy = (sharedConf[idxConfBlock(tx, ty, tz, 3)] - sharedConf[idxConfBlock(tx, ty - 1, tz, 3)]);
    dAyzdy = (sharedConf[idxConfBlock(tx, ty, tz, 4)] - sharedConf[idxConfBlock(tx, ty - 1, tz, 4)]);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    dAxzdz = (sharedConf[idxConfBlock(tx, ty, tz + 1, 2)] - sharedConf[idxConfBlock(tx, ty, tz, 2)]);
    dAyzdz = (sharedConf[idxConfBlock(tx, ty, tz + 1, 4)] - sharedConf[idxConfBlock(tx, ty, tz, 4)]);
    dAzzdz = (sharedConf[idxConfBlock(tx, ty, tz + 1, 5)] - sharedConf[idxConfBlock(tx, ty, tz, 5)]);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    dAxzdz = (sharedConf[idxConfBlock(tx, ty, tz, 2)] - sharedConf[idxConfBlock(tx, ty, tz - 1, 2)]);
    dAyzdz = (sharedConf[idxConfBlock(tx, ty, tz, 4)] - sharedConf[idxConfBlock(tx, ty, tz - 1, 4)]);
    dAzzdz = (sharedConf[idxConfBlock(tx, ty, tz, 5)] - sharedConf[idxConfBlock(tx, ty, tz - 1, 5)]);
}
