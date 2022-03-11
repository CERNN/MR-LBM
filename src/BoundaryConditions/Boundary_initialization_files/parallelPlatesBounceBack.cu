#include "boundaryCondition.cuh"

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