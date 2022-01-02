#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionPop(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop, dfloat *s_pop)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    //  BOUNDARY CONDITIONS SET TO LID DRIVEN CAVITY
    // Z - DIRECTION : PERIODIC
    // Y = NY-1 : LID MOVING IN THE +X DIRECTION
    // Y = 0 : BOUNCE BACK WALL
    // X - DIRECTION : BOUNCE BACK WALL

    //if (!(x == 0  || x == NX-1 || y == 0  || y == NY-1  || z == 0  || z == NZ-1 ))
    //    return;

    if (y == 0 && x == 0 && z == 0) // SWB
    {
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        #endif
        //Dead Pop are: [13, 14, 23, 24, 25, 26]
    }
    else if (y == 0 && x == 0 && z == (NZ_TOTAL - 1)) // SWF
    {
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        #endif
        //Dead Pop are: [13, 14, 23, 24, 25, 26]
    }
    else if (y == 0 && x == (NX - 1) && z == 0) // SEB
    {
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
        //Dead Pop are: [7, 8, 19, 20, 21, 22]
    }
    else if (y == 0 && x == (NX - 1) && z == (NZ_TOTAL - 1)) // SEF
    {
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
        //Dead Pop are: [7, 8, 19, 20, 21, 22]
    }
    else if (y == (NY - 1) && x == 0 && z == 0) // NWB
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[20] = pop[19] - 6 * rho_w * W3 *U_MAX         
        pop[22] = pop[21] - 6 * rho_w * W3 *U_MAX 
        #endif
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24 - 1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25 - 1)];
        #endif
    }
    else if (y == (NY - 1) && x == 0 && z == (NZ_TOTAL - 1)) // NWF
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[20] = pop[19] - 6 * rho_w * W3 *U_MAX         
        pop[22] = pop[21] - 6 * rho_w * W3 *U_MAX 
        #endif
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24 - 1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25 - 1)];
        #endif
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == 0) // NEB
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[23] = pop[24] + 6 * rho_w * W3 *U_MAX 
        pop[26] = pop[25] + 6 * rho_w * W3 *U_MAX 
        #endif
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19 - 1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == (NZ_TOTAL - 1)) // NEF
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[23] = pop[24] + 6 * rho_w * W3 *U_MAX 
        pop[26] = pop[25] + 6 * rho_w * W3 *U_MAX 
        #endif
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19 - 1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == 0 && x == 0) // SW
    {
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        #endif
        //Dead Pop are: [13, 14, 23, 24, 25, 26]
    }
    else if (y == 0 && x == (NX - 1)) // SE
    {
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
        //Dead Pop are: [7, 8, 19, 20, 21, 22]
    }
    else if (y == (NY - 1) && x == 0) // NW
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[20] = pop[19] - 6 * rho_w * W3 *U_MAX         
        pop[22] = pop[21] - 6 * rho_w * W3 *U_MAX 
        #endif

        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24 - 1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25 - 1)];
        #endif
    }
    else if (y == (NY - 1) && x == (NX - 1)) // NE
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[23] = pop[24] + 6 * rho_w * W3 *U_MAX 
        pop[26] = pop[25] + 6 * rho_w * W3 *U_MAX 
        #endif
        
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19 - 1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == 0 && z == 0) // SB
    {
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == 0 && z == (NZ_TOTAL - 1)) // SF
    {
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == (NY - 1) && z == 0) // NB
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[20] = pop[19] - 6 * rho_w * W3 *U_MAX         
        pop[22] = pop[21] - 6 * rho_w * W3 *U_MAX 
        pop[23] = pop[24] + 6 * rho_w * W3 *U_MAX 
        pop[26] = pop[25] + 6 * rho_w * W3 *U_MAX 
        #endif
    }
    else if (y == (NY - 1) && z == (NZ_TOTAL - 1)) // NF
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[20] = pop[19] - 6 * rho_w * W3 *U_MAX         
        pop[22] = pop[21] - 6 * rho_w * W3 *U_MAX 
        pop[23] = pop[24] + 6 * rho_w * W3 *U_MAX 
        pop[26] = pop[25] + 6 * rho_w * W3 *U_MAX 
        #endif
    }
    else if (x == 0 && z == 0) // WB
    {
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24 - 1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25 - 1)];
        #endif
    }
    else if (x == 0 && z == (NZ_TOTAL - 1)) // WF
    {
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24 - 1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25 - 1)];
        #endif
    }
    else if (x == (NX - 1) && z == 0) // EB
    {
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19 - 1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (x == (NX - 1) && z == (NZ_TOTAL - 1)) // EF
    {
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19 - 1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == 0) // S
    {
        pop[3] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 4 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[11] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[17] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 18 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (y == (NY - 1)) // N
    {
        #ifdef D3Q19
        const dfloat rho_w = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
        // f_i = f_i* w_i*rho_w*ci*u_w /c_s2
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        #endif
        #ifdef D3Q27
        const dfloat rho_w = rhoVar = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26];
        pop[ 4] = pop[3];
        pop[ 8] = pop[7] - 6 * rho_w * W2 * U_MAX;
        pop[12] = pop[11];
        pop[13] = pop[14] + 6 * rho_w * W2 * U_MAX;
        pop[18] = pop[17];
        pop[20] = pop[19] - 6 * rho_w * W3 *U_MAX         
        pop[22] = pop[21] - 6 * rho_w * W3 *U_MAX 
        pop[23] = pop[24] + 6 * rho_w * W3 *U_MAX 
        pop[26] = pop[25] + 6 * rho_w * W3 *U_MAX 
        #endif
    }
    else if (x == 0) // W
    {
        pop[1] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 2 - 1)];
        pop[7] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 8 - 1)];
        pop[9] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10 - 1)];
        pop[13] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14 - 1)];
        pop[15] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16 - 1)];
        #ifdef D3Q27
        pop[19] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 20 - 1)];
        pop[21] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 22 - 1)];
        pop[23] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 24 - 1)];
        pop[26] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 25 - 1)];
        #endif
    }
    else if (x == (NX - 1)) // E
    {
        pop[2] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 1 - 1)];
        pop[8] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 7 - 1)];
        pop[10] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 9 - 1)];
        pop[14] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13 - 1)];
        pop[16] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15 - 1)];
        #ifdef D3Q27
        pop[20] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 19 - 1)];
        pop[22] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 21 - 1)];
        pop[24] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 23 - 1)];
        pop[25] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 26 - 1)];
        #endif
    }
    else if (z == 0) // B
    {
        //PERIODIC
    }
    else if (z == (NZ_TOTAL - 1)) // F
    {
        //PERIODIC
    }
}