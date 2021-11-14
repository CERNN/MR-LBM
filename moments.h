#ifndef __MOMENTS_H
#define __MOMENTS_H

#include <builtin_types.h> // for device variables
#include "var.h"
#include "errorDef.h"


typedef struct moments {
private:
    int varLocation;
public:
    dfloat* rho;

    dfloat* ux;
    dfloat* uy;
    dfloat* uz;

    dfloat* pxx;
    dfloat* pxy;
    dfloat* pxz;
    dfloat* pyy;
    dfloat* pyz;
    dfloat* pzz;

    /* Constructor */
    __host__
        moments()
    {
        this->rho = nullptr;

        this->ux = nullptr;
        this->uy = nullptr;
        this->uz = nullptr;

        this->pxx = nullptr;
        this->pxy = nullptr;
        this->pxz = nullptr;
        this->pyy = nullptr;
        this->pyz = nullptr;
        this->pzz = nullptr;
    }

    /* Destructor */
    __host__
        ~moments()
    {
        this->rho = nullptr;

        this->ux = nullptr;
        this->uy = nullptr;
        this->uz = nullptr;

        this->pxx = nullptr;
        this->pxy = nullptr;
        this->pxz = nullptr;
        this->pyy = nullptr;
        this->pyz = nullptr;
        this->pzz = nullptr;
    }

    /* Allocate moments */
    __host__
        void momAllocation(int varLocation)
    {
        this->varLocation = varLocation;
        switch (varLocation)
        {
        case IN_HOST:
            // allocate with CUDA for pinned memory and for all GPUS
            checkCudaErrors(cudaMallocHost((void**)&(this->rho), MEM_SIZE_SCALAR));

            checkCudaErrors(cudaMallocHost((void**)&(this->ux), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->uz), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->uz), MEM_SIZE_SCALAR));

            checkCudaErrors(cudaMallocHost((void**)&(this->pxx), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->pxy), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->pxz), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->pyy), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->pyz), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocHost((void**)&(this->pzz), MEM_SIZE_SCALAR));

            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaMallocManaged((void**)&(this->rho), MEM_SIZE_SCALAR));

            checkCudaErrors(cudaMallocManaged((void**)&(this->ux), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->uy), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->uz), MEM_SIZE_SCALAR));

            checkCudaErrors(cudaMallocManaged((void**)&(this->pxx), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->pxy), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->pxz), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->pyy), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->pyz), MEM_SIZE_SCALAR));
            checkCudaErrors(cudaMallocManaged((void**)&(this->pzz), MEM_SIZE_SCALAR));
            
            break;
        default:
            break;
        }
    }

    /* Free moments */
    __host__
        void momFree()
    {
        switch (this->varLocation)
        {
        case IN_HOST:
            checkCudaErrors(cudaFreeHost(this->rho));

            checkCudaErrors(cudaFreeHost(this->ux));
            checkCudaErrors(cudaFreeHost(this->uy));
            checkCudaErrors(cudaFreeHost(this->uz));

            checkCudaErrors(cudaFreeHost(this->pxx));
            checkCudaErrors(cudaFreeHost(this->pxy));
            checkCudaErrors(cudaFreeHost(this->pxz));
            checkCudaErrors(cudaFreeHost(this->pyy));
            checkCudaErrors(cudaFreeHost(this->pyz));
            checkCudaErrors(cudaFreeHost(this->pzz));
            break;
        case IN_VIRTUAL:
            checkCudaErrors(cudaFree(this->rho));

            checkCudaErrors(cudaFree(this->ux));
            checkCudaErrors(cudaFree(this->uy));
            checkCudaErrors(cudaFree(this->uz));
            
            checkCudaErrors(cudaFree(this->pxx));
            checkCudaErrors(cudaFree(this->pxy));
            checkCudaErrors(cudaFree(this->pxz));
            checkCudaErrors(cudaFree(this->pyy));
            checkCudaErrors(cudaFree(this->pyz));
            checkCudaErrors(cudaFree(this->pzz));
            break;
        default:
            break;
        }
    }


} Moments;





#endif // !__MOMENTS_H