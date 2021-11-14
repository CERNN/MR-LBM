#ifndef __POPULATION_H
#define __POPULATION_H

#include <builtin_types.h> // for device variables
#include "var.h"
#include "errorDef.h"


typedef struct populations {
    private:
        int varLocation;
    public:
        dfloat* west;   // X = 0
        dfloat* east;   // X = NX
        dfloat* south;  // Y = 0
        dfloat* north;  // Y = NY 
        dfloat* back;   // Z = 0
        dfloat* front;  // Z = NZ
    
    /* Constructor */
    __host__
    populations()
    {
        this->west = nullptr;
        this->east = nullptr;

        this->south = nullptr;
        this->north = nullptr;

        this->back =  nullptr;
        this->front = nullptr;
    }

        /* Destructor */
    __host__
    ~populations()
    {
        this->west = nullptr;
        this->east = nullptr;

        this->south = nullptr;
        this->north = nullptr;

        this->back =  nullptr;
        this->front = nullptr;
    }
    /* Allocate populations */
    __host__
    void popAllocation(int varLocation)
    {
        this->varLocation = varLocation;
        switch (varLocation){
            case IN_HOST:
                checkCudaErrors(cudaMallocHost((void**)&(this->west), NY*NZ*QF*NUM_BLOCK_X));
                checkCudaErrors(cudaMallocHost((void**)&(this->east), NY*NZ*QF*NUM_BLOCK_X));

                checkCudaErrors(cudaMallocHost((void**)&(this->south), NX*NZ*QF*NUM_BLOCK_Y));
                checkCudaErrors(cudaMallocHost((void**)&(this->north), NX*NZ*QF*NUM_BLOCK_Y));

                checkCudaErrors(cudaMallocHost((void**)&(this->back),  NX*NY*QF*NUM_BLOCK_Z));
                checkCudaErrors(cudaMallocHost((void**)&(this->front), NX*NY*QF*NUM_BLOCK_Z));

                break;
            case IN_VIRTUAL:
                checkCudaErrors(cudaMallocManaged((void**)&(this->west), NY*NZ*QF*NUM_BLOCK_X));
                checkCudaErrors(cudaMallocManaged((void**)&(this->east), NY*NZ*QF*NUM_BLOCK_X));

                checkCudaErrors(cudaMallocManaged((void**)&(this->south), NX*NZ*QF*NUM_BLOCK_Y));
                checkCudaErrors(cudaMallocManaged((void**)&(this->north), NX*NZ*QF*NUM_BLOCK_Y));

                checkCudaErrors(cudaMallocManaged((void**)&(this->back),  NX*NY*QF*NUM_BLOCK_Z));
                checkCudaErrors(cudaMallocManaged((void**)&(this->front), NX*NY*QF*NUM_BLOCK_Z));
                break;
            default:
                break;
        }

    }
    /* Free populations */
    __host__
    void popFree()
    {
        switch (this->varLocation)
        {
        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->west));
            checkCudaErrors(cudaFreeHost(this->east));
            checkCudaErrors(cudaFreeHost(this->south));
            checkCudaErrors(cudaFreeHost(this->north));
            checkCudaErrors(cudaFreeHost(this->back));
            checkCudaErrors(cudaFreeHost(this->front));
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->west));
            checkCudaErrors(cudaFree(this->east));
            checkCudaErrors(cudaFree(this->south));
            checkCudaErrors(cudaFree(this->north));
            checkCudaErrors(cudaFree(this->back));
            checkCudaErrors(cudaFree(this->front));
            break;
        default:
            break;
        }
    }
}Populations;

#endif