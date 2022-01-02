#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionMom(
    dim3 threadIdx, dim3 blockIdx, dfloat *mom){
    /*
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (y == 0 && x == 0 && z == 0) // SWB
    {

    }
    else if (y == 0 && x == 0 && z == (NZ_TOTAL-1)) // SWF
    {

    }
    else if (y == 0 && x == (NX - 1) && z == 0) // SEB
    {

    }
    else if (y == 0 && x == (NX - 1) && z == (NZ_TOTAL-1)) // SEF
    {

    }
    else if (y == (NY - 1) && x == 0 && z == 0) // NWB
    {

    }
    else if (y == (NY - 1) && x == 0 && z == (NZ_TOTAL-1)) // NWF
    {

    }
    else if (y == (NY - 1) && x == (NX - 1) && z == 0) // NEB
    {

    }
    else if (y == (NY - 1) && x == (NX - 1) && z == (NZ_TOTAL-1)) // NEF
    {

    }
    else if (y == 0 && x == 0) // SW
    {

    }
    else if (y == 0 && x == (NX - 1)) // SE
    {

    }
    else if (y == (NY - 1) && x == 0) // NW
    {

    }
    else if (y == (NY - 1) && x == (NX - 1)) // NE
    {

    }
    else if (y == 0 && z == 0) // SB
    {

    }
    else if (y == 0 && z == (NZ_TOTAL-1)) // SF
    {

    }
    else if (y == (NY - 1) && z == 0) // NB
    {

    }
    else if (y == (NY - 1) && z == (NZ_TOTAL-1)) // NF
    {

    }
    else if (x == 0 && z == 0) // WB
    {
    }
    else if (x == 0 && z == (NZ_TOTAL-1)) // WF
    {
    }
    else if (x == (NX - 1) && z == 0) // EB
    {
    }
    else if (x == (NX - 1) && z == (NZ_TOTAL-1)) // EF
    {
    }
    else if (y == 0) // S
    {

    }
    else if (y == (NY - 1)) // N
    {

    }
    else if (x == 0) // W
    {
    }
    else if (x == (NX - 1)) // E
    {
    }
    else if (z == 0) // B
    {
    }
    else if (z == (NZ_TOTAL-1)) // F
    {  
    }
*/
}