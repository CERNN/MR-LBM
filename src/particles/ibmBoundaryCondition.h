/*
*   @file ibmVar.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @brief Configurations for the boundary conditions that affect the immersed boundary method
*   @version 0.3.0
*   @date 14/06/2021
*/

#ifndef __IBM_BC_H
#define __IBM_BC_H

#include "../../var.h"
#include "ibmVar.h"
#include <stdio.h>
#include <math.h>

/* -------------------------- BOUNDARY CONDITIONS -------------------------- */


// era para definir se tem parede ou nao -> periodico ou nao
// --- X direction ---
//#define IBM_BC_X_WALL



#ifdef IBM_BC_X_WALL
    // TODO: not implemented yet
    #define IBM_BC_X_WALL_UY 0.0
    #define IBM_BC_X_WALL_UZ 0.0
#endif //IBM_BC_X_WALL

#ifdef IBM_BC_X_PERIODIC
    #define IBM_BC_X_0 (0)
    #define IBM_BC_X_E (NX-0)
#endif //IBM_BC_X_PERIODIC



//#define IBM_BC_Y_PERIODIC

#ifdef IBM_BC_Y_WALL
    // TODO: not implemented yet
    #define IBM_BC_Y_WALL_UX 0.0
    #define IBM_BC_Y_WALL_UZ 0.0
#endif //IBM_BC_Y_WALL

#ifdef IBM_BC_Y_PERIODIC
    #define IBM_BC_Y_0 0
    #define IBM_BC_Y_E (NY-0)
#endif //IBM_BC_Y_PERIODIC



// --- Z direction ---
//#define IBM_BC_Z_WALL


#ifdef IBM_BC_Z_WALL
    // TODO: not implemented yet
    #define IBM_BC_Z_WALL_UX 0.0
    #define IBM_BC_Z_WALL_UY 0.0
#endif //IBM_BC_Z_WALL

#ifdef IBM_BC_Z_PERIODIC
    //TODO: FIX with multi-gpu, it currently does not work with values different than 0 and NZ_TOTAl
    #define IBM_BC_Z_0 0
    #define IBM_BC_Z_E (NZ_TOTAL-0)
#endif //IBM_BC_Z_PERIODIC



#endif // !__IBM_BC_H