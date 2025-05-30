/*
*   @file particleCenter.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Ricardo de Souza (rsouza.1996@alunos.utfpr.edu.br)
*   @brief Class for IBM particle center
*   @version 0.3.0
*   @date 19/05/2025
*/

#include "IbmMacrsAux.hpp"

__host__
IbmMacrsAux::IbmMacrsAux() {
}

__host__
IbmMacrsAux::~IbmMacrsAux() {
}

__host__
void IbmMacrsAux::ibmMacrsAuxAllocation() {
    for (int i = 0; i < N_GPUS; i++) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        velAux[i].allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
        fAux[i].allocateMemory(NUMBER_LBM_IB_MACR_NODES, IN_VIRTUAL);
    }
}

__host__
void IbmMacrsAux::ibmMacrsAuxFree() {
    for (int i = 0; i < N_GPUS; i++) {
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        velAux[i].freeMemory();
        fAux[i].freeMemory();
    }
}
