#ifndef __IBM_MACRS_AUX_H
#define __IBM_MACRS_AUX_H

#include "ibmVar.h"
#include "../../../globalFunctions.h"

/*
*   Class for LBM macroscopics (auxiliary IBM)
*/
class IbmMacrsAux
{
private:
    dfloat3SoA velAux[N_GPUS];  // auxiliary velocities
    dfloat3SoA fAux[N_GPUS];    // auxiliary forces, for synchronization

public:
    /* Constructor */
    __host__ IbmMacrsAux();

    /* Destructor */
    __host__ ~IbmMacrsAux();

    /* Getters */
    __host__ __device__ dfloat3SoA& getVelAux(int gpu);
    __host__ __device__ dfloat3SoA& getFAux(int gpu);

    /* Setters */
    __host__ __device__ void setVelAux(int gpu, const dfloat3SoA& value);
    __host__ __device__ void setFAux(int gpu, const dfloat3SoA& value);

    /* Allocate IBM macroscopics aux */
    __host__ void ibmMacrsAuxAllocate();

    /* Free IBM macroscopics aux */
    __host__ void ibmMacrsAuxFree();
};

#endif // !__IBM_MACRS_AUX_H
