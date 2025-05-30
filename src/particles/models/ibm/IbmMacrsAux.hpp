/*
*   @file particleCenter.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Ricardo de Souza (rsouza.1996@alunos.utfpr.edu.br)
*   @brief Class for IBM particle center
*   @version 0.3.0
*   @date 19/05/2025
*/

#ifndef __IBM_MACRS_AUX_H
#define __IBM_MACRS_AUX_H

#include "../models/ibm/ibmVar.h"
#include "../../globalFunctions.h"

/*
*   Class for LBM macroscopics
*/
class IbmMacrsAux
{
public:
    /* Constructor */
    __host__
    IbmMacrsAux();

    /* Destructor */
    __host__
    ~IbmMacrsAux();


    /* Allocate IBM macroscopics aux */
    __host__
    void ibmMacrsAuxAllocation();

    /* Free IBM macroscopics aux */
    __host__
    void ibmMacrsAuxFree();
    
protected:
    dfloat3SoA velAux[N_GPUS];  // auxiliary velocities
    dfloat3SoA fAux[N_GPUS]; // auxiliary forces, for synchronization

};

#endif // !__IBM_MACRS_AUX_H