#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionPop(
    dim3 threadIdx, dim3 blockIdx, dfloat *pop,dfloat *s_pop, char dNodeType){

}

__global__ void gpuInitialization_nodeType(
    char *dNodeType)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = BULK;

}