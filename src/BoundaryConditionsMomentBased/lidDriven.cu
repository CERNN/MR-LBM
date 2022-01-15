#include "boundaryCondition.cuh"


__device__ void gpuBoundaryConditionMom(
    dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz ){


    //  BOUNDARY CONDITIONS SET TO LID DRIVEN CAVITY
    // Z - DIRECTION : PERIODIC
    // Y = NY-1 : LID MOVING IN THE +X DIRECTION
    // Y = 0 : BOUNCE BACK WALL
    // X - DIRECTION : BOUNCE BACK WALL



    switch (dNodeType){
        case BULK:
            break;
        //corners
        case SOUTH_WEST_BACK:
            gpuBCMomentSW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_WEST_FRONT:
            gpuBCMomentSW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_EAST_BACK:
            gpuBCMomentSE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_EAST_FRONT:
            gpuBCMomentSE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_WEST_BACK://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentNW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_WEST_FRONT://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentNW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_EAST_BACK://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentNE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_EAST_FRONT://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentNE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;


        //edges
        case NORTH_WEST://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentNW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_EAST://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentNE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_FRONT://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentN(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case NORTH_BACK://TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentN(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_EAST:
            gpuBCMomentSE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_WEST:
            gpuBCMomentSW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_FRONT:
            gpuBCMomentS(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case SOUTH_BACK:
            gpuBCMomentS(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case WEST_FRONT:
            gpuBCMomentW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case WEST_BACK:
            gpuBCMomentW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case EAST_FRONT:
            gpuBCMomentE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case EAST_BACK:
            gpuBCMomentE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;


        // face
        case SOUTH:
            gpuBCMomentS(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;

        case NORTH: //TODO: this ones have velocity, need make a way to pass the velocity index
            gpuBCMomentN(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;

        case EAST:
            gpuBCMomentE(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
            break;
        case WEST:
            gpuBCMomentW(pop,rhoVar,dNodeType,uxVar,uyVar,uzVar,pixx,pixy,pixz,piyy,piyz,pizz);
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