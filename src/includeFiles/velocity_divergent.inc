
//define shared memory
__shared__ dfloat sharedVel[VEL_GRAD_BLOCK_SIZE];

sharedVel[idxVelBlock(tx, ty, tz, 0)] = ux_t30;
sharedVel[idxVelBlock(tx, ty, tz, 1)] = uy_t30;
sharedVel[idxVelBlock(tx, ty, tz, 2)] = uz_t30;

// Load neighboring halo cells in shared memory
#include "includeFiles\velLoad.inc"
__syncthreads();

// Compute unrolled divergent calculations
//FINITE CENTRAL DIFFERENCE ORDER 1, otherwise i would have to store a halo with width of 2 in shared memory
dfloat duxdx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 0)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 0)]) / 2;
dfloat duxdy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 0)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 0)]) / 2;
dfloat duxdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 0)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 0)]) / 2;

dfloat duydx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 1)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 1)]) / 2;
dfloat duydy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 1)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 1)]) / 2;
dfloat duydz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 1)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 1)]) / 2;

dfloat duzdx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 2)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 2)]) / 2;
dfloat duzdy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 2)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 2)]) / 2;
dfloat duzdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 2)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 2)]) / 2;

//OVERWRITE FOR NEAR WALL NODES
if ((nodeType & 0b01010101) == 0b01010101) { // wall west
    duxdx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 0)] - sharedVel[idxVelBlock(tx, ty, tz, 0)]);
    duydx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 1)] - sharedVel[idxVelBlock(tx, ty, tz, 1)]);
    duzdx_t30 = (sharedVel[idxVelBlock(tx + 1, ty, tz, 2)] - sharedVel[idxVelBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b10101010) == 0b10101010) { // wall east
    duxdx_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 0)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 0)]);
    duydx_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 1)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 1)]);
    duzdx_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 2)] - sharedVel[idxVelBlock(tx - 1, ty, tz, 2)]);
}
if ((nodeType & 0b00110011) == 0b00110011) { // wall south
    duxdy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 0)] - sharedVel[idxVelBlock(tx, ty, tz, 0)]);
    duydy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 1)] - sharedVel[idxVelBlock(tx, ty, tz, 1)]);
    duzdy_t30 = (sharedVel[idxVelBlock(tx, ty + 1, tz, 2)] - sharedVel[idxVelBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b11001100) == 0b11001100) { // wall north
    duxdy_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 0)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 0)]);
    duydy_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 1)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 1)]);
    duzdy_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 2)] - sharedVel[idxVelBlock(tx, ty - 1, tz, 2)]);
}
if ((nodeType & 0b11110000) == 0b11110000) { // wall front
    duxdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 0)] - sharedVel[idxVelBlock(tx, ty, tz, 0)]);
    duydz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 1)] - sharedVel[idxVelBlock(tx, ty, tz, 1)]);
    duzdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz + 1, 2)] - sharedVel[idxVelBlock(tx, ty, tz, 2)]);
}
if ((nodeType & 0b00001111) == 0b00001111) { // wall back
    duxdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 0)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 0)]);
    duydz_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 1)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 1)]);
    duzdz_t30 = (sharedVel[idxVelBlock(tx, ty, tz, 2)] - sharedVel[idxVelBlock(tx, ty, tz - 1, 2)]);
}