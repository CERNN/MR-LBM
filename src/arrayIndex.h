#ifndef __ARRAYINDEX_H
#define __ARRAYINDEX_H

#include "var.h"
//#include "definitions.h"


constexpr int M_RHO_INDEX = 0;
constexpr int M_UX_INDEX  = 1;
constexpr int M_UY_INDEX  = 2;
constexpr int M_UZ_INDEX  = 3;
constexpr int M_MXX_INDEX = 4;
constexpr int M_MXY_INDEX = 5;
constexpr int M_MXZ_INDEX = 6;
constexpr int M_MYY_INDEX = 7;
constexpr int M_MYZ_INDEX = 8;
constexpr int M_MZZ_INDEX = 9;

#ifdef M_OFFSET
    #undef M_OFFSET
#endif
#define M_OFFSET M_MZZ_INDEX

#ifdef NON_NEWTONIAN_FLUID
    constexpr int M_OMEGA_INDEX = (1+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET M_OMEGA_INDEX
#endif

#ifdef LOCAL_FORCES
    constexpr int M_FX_INDEX = (1+M_OFFSET);
    constexpr int M_FY_INDEX = (2+M_OFFSET);
    constexpr int M_FZ_INDEX = (3+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET M_FZ_INDEX
#endif

#ifdef SECOND_DIST
    constexpr int M_C_INDEX   = (1+M_OFFSET);
    constexpr int M_CX_INDEX  = (2+M_OFFSET);
    constexpr int M_CY_INDEX  = (3+M_OFFSET);
    constexpr int M_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET M_CZ_INDEX
#endif


const size_t NUMBER_MOMENTS = M_OFFSET+1;

__device__ const dfloat MOMENT_SCALE[10]   =  {1, 3, 3, 3, 4.5, 9.0, 9.0, 4.5, 9.0, 4.5};
__device__ const dfloat MOMENT_OFFSENT[10] =  {RHO_0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


#endif