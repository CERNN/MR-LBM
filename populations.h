#ifndef __POPULATION_H
#define __POPULATION_H

#include <builtin_types.h> // for device variables
#include "var.h"
#include "errorDef.h"


typedef struct populations {
    private:
        int varLocation;
    public:
        dfloat* x;   // X = 0
        dfloat* y;   // X = NX
        dfloat* z;  // Y = 0

    /* Constructor */
    __host__
    populations()
    {
        this->x = nullptr;
        this->y = nullptr;
        this->z = nullptr;

    }

        /* Destructor */
    __host__
    ~populations()
    {
        this->x = nullptr;
        this->y = nullptr;
        this->z = nullptr;

    }
    /* Allocate populations */
    __host__
    void popAllocation(int varLocation)
    {
        this->varLocation = varLocation;
        switch (varLocation){
            case IN_HOST:
                checkCudaErrors(cudaMallocHost((void**)&(this->x), NUMBER_GHOST_FACE_YZ*QF*sizeof(dfloat)));
                checkCudaErrors(cudaMallocHost((void**)&(this->y), NUMBER_GHOST_FACE_XZ*QF*sizeof(dfloat)));
                checkCudaErrors(cudaMallocHost((void**)&(this->z), NUMBER_GHOST_FACE_XY*QF*sizeof(dfloat)));

                break;
            case IN_VIRTUAL:
                checkCudaErrors(cudaMallocManaged((void**)&(this->x), NUMBER_GHOST_FACE_YZ*QF*sizeof(dfloat)));
                checkCudaErrors(cudaMallocManaged((void**)&(this->y), NUMBER_GHOST_FACE_XZ*QF*sizeof(dfloat)));
                checkCudaErrors(cudaMallocManaged((void**)&(this->z), NUMBER_GHOST_FACE_XY*QF*sizeof(dfloat)));

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
            checkCudaErrors(cudaFreeHost(this->x));
            checkCudaErrors(cudaFreeHost(this->y));
            checkCudaErrors(cudaFreeHost(this->z));
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->y));
            checkCudaErrors(cudaFree(this->y));
            checkCudaErrors(cudaFree(this->z));
            break;
        default:
            break;
        }
    }
}Populations;

#endif