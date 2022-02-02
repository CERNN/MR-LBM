#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionPop(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop,dfloat *s_pop, char dNodeType){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    //  BOUNDARY CONDITIONS SET TO PARALLEL PLATES NORTH AND SOUTH

    //if (!(x == 0  || x == NX-1 || y == 0  || y == NY-1  || z == 0  || z == NZ-1 ))
    //    return;
    
    switch (dNodeType){
        case BULK:
            break;
        case SOUTH:
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
            break;

        case NORTH:
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

    if (y == 0){ //S
        nodeType = SOUTH;
    }else if (y == (NY - 1)){ // N
        nodeType = NORTH;
    }else{
        nodeType = BULK;
    }
    
    dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;

}