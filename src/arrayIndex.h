#ifndef __ARRAYINDEX_H
#define __ARRAYINDEX_H

#include "var.h"
#include "definitions.h"



#define M_RHO_INDEX  0
#define M_UX_INDEX   1
#define M_UY_INDEX   2
#define M_UZ_INDEX   3
#define M_MXX_INDEX  4
#define M_MXY_INDEX  5
#define M_MXZ_INDEX  6
#define M_MYY_INDEX  7
#define M_MYZ_INDEX  8
#define M_MZZ_INDEX  9

/*
#ifdef M_OFFSET
    #undef M_OFFSET
#endif
#define M_OFFSET M_MZZ_INDEX

#ifdef NON_NEWTONIAN_FLUID
    #define M_OMEGA_INDEX (1+M_OFFSET)
    #undef M_OFFSET
    #define M_OFFSET M_OMEGA_INDEX
#endif

#ifdef LOCAL_FORCES
    #define M_FX_INDEX (1+M_OFFSET)
    #define M_FY_INDEX (2+M_OFFSET)
    #define M_FZ_INDEX (3+M_OFFSET)

    #undef M_OFFSET
    #define M_OFFSET M_FZ_INDEX
#endif
*/



#endif