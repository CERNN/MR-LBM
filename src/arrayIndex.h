/**
 *  @file arrayIndex.h
 *  Contributors history:
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Array index for fMom
 *  @version 0.1.0
 *  @date 01/09/2025
 */


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

#ifdef OMEGA_FIELD
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
    constexpr int M2_C_INDEX   = (1+M_OFFSET);
    constexpr int M2_CX_INDEX  = (2+M_OFFSET);
    constexpr int M2_CY_INDEX  = (3+M_OFFSET);
    constexpr int M2_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET M2_CZ_INDEX
#endif

#ifdef A_XX_DIST
    constexpr int A_XX_C_INDEX   = (1+M_OFFSET);
    //constexpr int G_XX_C_INDEX   = (2+M_OFFSET);
    constexpr int A_XX_CX_INDEX  = (2+M_OFFSET);
    constexpr int A_XX_CY_INDEX  = (3+M_OFFSET);
    constexpr int A_XX_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET A_XX_CZ_INDEX
#endif

#ifdef A_XY_DIST
    constexpr int A_XY_C_INDEX   = (1+M_OFFSET);
    //constexpr int G_XY_C_INDEX   = (2+M_OFFSET);
    constexpr int A_XY_CX_INDEX  = (2+M_OFFSET);
    constexpr int A_XY_CY_INDEX  = (3+M_OFFSET);
    constexpr int A_XY_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET A_XY_CZ_INDEX
#endif

#ifdef A_XZ_DIST
    constexpr int A_XZ_C_INDEX   = (1+M_OFFSET);
    //constexpr int G_XZ_C_INDEX   = (2+M_OFFSET);
    constexpr int A_XZ_CX_INDEX  = (2+M_OFFSET);
    constexpr int A_XZ_CY_INDEX  = (3+M_OFFSET);
    constexpr int A_XZ_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET A_XZ_CZ_INDEX
#endif

#ifdef A_YY_DIST
    constexpr int A_YY_C_INDEX   = (1+M_OFFSET);
    //constexpr int G_YY_C_INDEX   = (2+M_OFFSET);
    constexpr int A_YY_CX_INDEX  = (2+M_OFFSET);
    constexpr int A_YY_CY_INDEX  = (3+M_OFFSET);
    constexpr int A_YY_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET A_YY_CZ_INDEX
#endif

#ifdef A_YZ_DIST
    constexpr int A_YZ_C_INDEX   = (1+M_OFFSET);
    //constexpr int G_YZ_C_INDEX   = (2+M_OFFSET);
    constexpr int A_YZ_CX_INDEX  = (2+M_OFFSET);
    constexpr int A_YZ_CY_INDEX  = (3+M_OFFSET);
    constexpr int A_YZ_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET A_YZ_CZ_INDEX
#endif

#ifdef A_ZZ_DIST
    constexpr int A_ZZ_C_INDEX   = (1+M_OFFSET);
    //constexpr int G_ZZ_C_INDEX   = (2+M_OFFSET);
    constexpr int A_ZZ_CX_INDEX  = (2+M_OFFSET);
    constexpr int A_ZZ_CY_INDEX  = (3+M_OFFSET);
    constexpr int A_ZZ_CZ_INDEX  = (4+M_OFFSET);
    #ifdef M_OFFSET
        #undef M_OFFSET
    #endif
    #define M_OFFSET A_ZZ_CZ_INDEX
#endif

#ifdef LOG_CONFORMATION
    #ifdef A_XX_DIST
        constexpr int C_XX_1_INDEX = (1 + M_OFFSET);
        constexpr int C_XX_2_INDEX = (2 + M_OFFSET);
        #ifdef M_OFFSET
            #undef M_OFFSET
        #endif
        #define M_OFFSET C_XX_2_INDEX
    #endif

    #ifdef A_XY_DIST
        constexpr int C_XY_1_INDEX = (1 + M_OFFSET);
        constexpr int C_XY_2_INDEX = (2 + M_OFFSET);
        #ifdef M_OFFSET
            #undef M_OFFSET
        #endif
        #define M_OFFSET C_XY_2_INDEX
    #endif

    #ifdef A_XZ_DIST
        constexpr int C_XZ_1_INDEX = (1 + M_OFFSET);
        constexpr int C_XZ_2_INDEX = (2 + M_OFFSET);
        #ifdef M_OFFSET
            #undef M_OFFSET
        #endif
        #define M_OFFSET C_XZ_2_INDEX
    #endif

    #ifdef A_YY_DIST
        constexpr int C_YY_1_INDEX = (1 + M_OFFSET);
        constexpr int C_YY_2_INDEX = (2 + M_OFFSET);
        #ifdef M_OFFSET
            #undef M_OFFSET
        #endif
        #define M_OFFSET C_YY_2_INDEX
    #endif

    #ifdef A_YZ_DIST
        constexpr int C_YZ_1_INDEX = (1 + M_OFFSET);
        constexpr int C_YZ_2_INDEX = (2 + M_OFFSET);
        #ifdef M_OFFSET
            #undef M_OFFSET
        #endif
        #define M_OFFSET C_YZ_2_INDEX
    #endif

    #ifdef A_ZZ_DIST
        constexpr int C_ZZ_1_INDEX = (1 + M_OFFSET);
        constexpr int C_ZZ_2_INDEX = (2 + M_OFFSET);
        #ifdef M_OFFSET
            #undef M_OFFSET
        #endif
        #define M_OFFSET C_ZZ_2_INDEX
    #endif
#endif




const size_t NUMBER_MOMENTS = M_OFFSET+1;

__device__ const dfloat MOMENT_SCALE[10]   =  {1, 3, 3, 3, 4.5, 9.0, 9.0, 4.5, 9.0, 4.5};
__device__ const dfloat MOMENT_OFFSENT[10] =  {RHO_0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


#endif