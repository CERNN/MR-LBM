#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionPop(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop, dfloat *s_pop, char dNodeType){
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

    switch (dNodeType){
        case BULK:
            break;
        //corners
        case SOUTH_WEST_BACK:
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
                break;
        case SOUTH_WEST_FRONT:
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
            break;
        case SOUTH_EAST_BACK:
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
            break;
        case SOUTH_EAST_FRONT:
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
            break;
        case NORTH_WEST_BACK://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;
        case NORTH_WEST_FRONT://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;
        case NORTH_EAST_BACK://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;
        case NORTH_EAST_FRONT://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;


        //edges
        case NORTH_WEST://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;
        case NORTH_EAST://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;
        case NORTH_FRONT://TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;
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
            break;
        case SOUTH_EAST:
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
            break;
        case SOUTH_WEST:
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
            break;
        case SOUTH_FRONT:
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
            break;
        case SOUTH_BACK:
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
            break;
        case WEST_FRONT:
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
            break;
        case WEST_BACK:
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
            break;
        case EAST_FRONT:
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
            break;
        case EAST_BACK:
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
            break;


        // face
        case SOUTH:
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
            break;

        case NORTH: //TODO: this ones have velocity, need make a way to pass the velocity index
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
            break;

        case EAST:
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
            break;
        case WEST:
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
            break;
        //periodic
        case BACK:
            break;
        case FRONT:
            break;
        default:
            break;
    }
}

__global__ void gpuInitialization_nodeType(
    char *dNodeType)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
    char nodeType;


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
        nodeType = NORTH_WEST_BACK;
    }
    else if (y == (NY - 1) && x == 0 && z == (NZ_TOTAL - 1)) // NWF
    {
        nodeType = NORTH_WEST_FRONT;
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == 0) // NEB
    {
        nodeType = NORTH_EAST_BACK;
    }
    else if (y == (NY - 1) && x == (NX - 1) && z == (NZ_TOTAL - 1)) // NEF
    {
        nodeType = NORTH_EAST_FRONT;
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
         nodeType = NORTH_WEST;
    }
    else if (y == (NY - 1) && x == (NX - 1)) // NE
    {
        nodeType = NORTH_EAST;
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
        nodeType = NORTH_BACK;
    }
    else if (y == (NY - 1) && z == (NZ_TOTAL - 1)) // NF
    {
        nodeType = NORTH_FRONT;
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
        nodeType = EAST_BACK
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
        nodeType = BULK;
    }
    else if (z == (NZ_TOTAL - 1)) // F
    {
        nodeType = BULK;      
    }
    else{
        nodeType = BULK;
    }
    
    dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;

}