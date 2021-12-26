#include "boundaryCondition.cuh"

__device__ void gpuBoundaryCondition(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop,dfloat *s_pop){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if !(x == 0  || x == NX-1 || y == 0  || y == NY-1  || z == 0  || z == NZ-1 )
        return;


    if (y == 0 && x == 0 && z == 0) // SWB
    {
        pop[ 1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2-1)];
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6-1)];
        pop[ 7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8-1)];
        pop[ 9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10-1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12-1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20-1)];
        #endif
        //Dead Pop are: [13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
    }
    else if (y == 0 && x == 0 && z == (NZ_TOTAL-1)) // SWF
    {
        pop[ 1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2-1)];
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5-1)];
        pop[ 7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8-1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16-1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18-1)];
        #ifdef D3Q27
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22-1)];
        #endif
        //Dead Pop are: [9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26]
    }
    else if (y == 0 && x == (NX - 1) && z == 0) // SEB
    {
        pop[ 2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1-1)];
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6-1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12-1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13-1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15-1)];
        #ifdef D3Q27
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26-1)];
        #endif
        //Dead Pop are: [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
    }
    else if (y == 0 && x == (NX - 1) && z == (NZ_TOTAL-1)) // SEF
    {
        pop[ 2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1-1)];
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5-1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9-1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13-1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18-1)];
        #ifdef D3Q27
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23-1)];
        #endif
        //Dead Pop are: [7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 25, 26]
    }
    else if (y == (NY - 1) && x == 0 && z == 0) // NWB
    {
        pop[ 1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2-1)];
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6-1)];
        pop[ 9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10-1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14-1)];
        pop[18] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17-1)];
        #ifdef D3Q27
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24-1)];
        #endif
        //Dead Pop are: [7, 8, 11, 12, 15, 16, 19, 20, 21, 22, 25, 26]
    }
    else if (y == (NY - 1) && x == 0 && z == (NZ_TOTAL-1)) // NWF
    {
        pop[ 1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2-1)];
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5-1)];
        pop[12] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11-1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14-1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16-1)];
        #ifdef D3Q27
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25-1)];
        #endif
        //Dead Pop are: [7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23, 24]
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == 0) // NEB
    {
        pop[ 2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1-1)];
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6-1)];
        pop[ 8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7-1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15-1)];
        pop[18] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17-1)];
        #ifdef D3Q27
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21-1)];
        #endif
        //Dead Pop are: [9, 10, 11, 12, 13, 14, 19, 20, 23, 24, 25, 26]
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == (NZ_TOTAL-1)) // NEF
    {
        pop[ 2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1-1)];
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5-1)];
        pop[ 8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7-1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9-1)];
        pop[12] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11-1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19-1)];
        #endif
        //Dead Pop are: [13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26]
    }
    else if (y == 0 && x == 0) // SW
    {
        pop[ 1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2-1)];
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8-1)];
        pop[ 9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10-1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12-1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16-1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18-1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20-1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22-1)];
        #endif
        //Dead Pop are: [13, 14, 23, 24, 25, 26]
    }
    else if (y == 0 && x == (NX - 1)) // SE
    {
        pop[ 2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1-1)];
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9-1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12-1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13-1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15-1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18-1)];
        #ifdef D3Q27
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23-1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26-1)];
        #endif
        //Dead Pop are: [7, 8, 19, 20, 21, 22]
    }
    else if (y == (NY - 1) && x == 0) // NW
    {
        pop[ 1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2-1)];
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10-1)];
        pop[12] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11-1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14-1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16-1)];
        pop[18] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17-1)];
        #ifdef D3Q27
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24-1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25-1)];
        #endif
        //Dead Pop are: [7, 8, 19, 20, 21, 22]
    }
    else if (y == (NY - 1) && x == (NX - 1)) // NE
    {
        pop[ 2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1-1)];
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7-1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9-1)];
        pop[12] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11-1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15-1)];
        pop[18] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17-1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19-1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21-1)];
        #endif
        //Dead Pop are: [13, 14, 23, 24, 25, 26]
    }
    else if (y == 0 && z == 0) // SB
    {
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6-1)];
        pop[ 7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8-1)];
        pop[ 9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10-1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12-1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13-1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15-1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20-1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26-1)];
        #endif
        //Dead Pop are: [17, 18, 21, 22, 23, 24]
    }
    else if (y == 0 && z == (NZ_TOTAL-1)) // SF
    {
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5-1)];
        pop[ 7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8-1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9-1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13-1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16-1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18-1)];
        #ifdef D3Q27
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22-1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23-1)];
        #endif
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
    }
    else if (y == (NY - 1) && z == 0) // NB
    {
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 6-1)];
        pop[ 8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7-1)];
        pop[ 9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10-1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14-1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15-1)];
        pop[18] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17-1)];
        #ifdef D3Q27
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21-1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24-1)];
        #endif
    //Dead Pop are: [11, 12, 19, 20, 25, 26]
    }
    else if (y == (NY - 1) && z == (NZ_TOTAL-1)) // NF
    {
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 5-1)];
        pop[ 8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7-1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9-1)];
        pop[12] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11-1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14-1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16-1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19-1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25-1)];
        #endif
    //Dead Pop are: [17, 18, 21, 22, 23, 24]
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
        pop[ 3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4-1)];
        pop[ 7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8-1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12-1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13-1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18-1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20-1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22-1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23-1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26-1)];
        #endif
    }
    else if (y == (NY - 1)) // N
    {
        pop[ 4] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 3-1)];
        pop[ 8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7-1)];
        pop[12] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11-1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14-1)];
        pop[18] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17-1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19-1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21-1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24-1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25-1)];
        #endif
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

}