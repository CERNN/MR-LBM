#include "boundaryCondition.cuh"


__device__ void gpuBoundaryConditionMom(
    dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz ){

    /*
    * uVar = 3u
    * pixx = 4.5pixx
    * pixy = 9.0pixy
    */


    switch (dNodeType){
        case BULK:
            break;
        case SOUTH:
            gpuBCMomentS(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;

        case NORTH:
            gpuBCMomentN(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
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

    if (y == 0){ //S
        nodeType = SOUTH;
    }else if (y == (NY - 1)){ // N
        nodeType = NORTH;
    }else{
        nodeType = BULK;
    }
    
    dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;

}