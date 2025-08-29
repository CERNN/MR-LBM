#include "IbmMacrsAux.cuh"

/* Constructor */
__host__ IbmMacrsAux::IbmMacrsAux(){}

/* Destructor */
__host__ IbmMacrsAux::~IbmMacrsAux(){}

/* Getters */
__host__ __device__ dfloat3SoA& IbmMacrsAux::getVelAux(int gpu)
{
    return velAux[gpu];
}

__host__ __device__ dfloat3SoA& IbmMacrsAux::getFAux(int gpu)
{
    return fAux[gpu];
}

/* Setters */
__host__ __device__ void IbmMacrsAux::setVelAux(int gpu, const dfloat3SoA& value)
{
    velAux[gpu] = value;
}

__host__ __device__ void IbmMacrsAux::setFAux(int gpu, const dfloat3SoA& value)
{
    fAux[gpu] = value;
}

/* Allocate IBM macroscopics aux */
__host__ void IbmMacrsAux::ibmMacrsAuxAllocate()
{
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        this->velAux[i].allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
        this->fAux[i].allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
    }
}

/* Free IBM macroscopics aux */
__host__ void IbmMacrsAux::ibmMacrsAuxFree()
{
    for(int i = 0; i < N_GPUS; i++)
    {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        this->velAux[i].freeMemory();
        this->fAux[i].freeMemory();
    }
}
