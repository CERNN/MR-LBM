/**
 *  @file var.h
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Global variables
 *  @version 0.4.0
 *  @date 01/09/2025
 */


#ifndef __VAR_H
#define __VAR_H


#include <builtin_types.h>  // for devices variables
#include <stdint.h>         // for uint32_t
#include <cstdint>

#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

#define SINGLE_PRECISION 
#ifdef SINGLE_PRECISION
    typedef float dfloat;      // single precision
    #define VTK_DFLOAT_TYPE "float"
#endif //SINGLE_PRECISION
#ifdef DOUBLE_PRECISION
    typedef double dfloat;      // double precision
    #define VTK_DFLOAT_TYPE "double"
#endif //DOUBLE_PRECISION

// Pow function to use
#ifdef SINGLE_PRECISION
    #define POW_FUNCTION powf 
#else
    #define POW_FUNCTION pow
#endif //SINGLE_PRECISION


#define CELLDATA_SAVE false

/* ----------------------------- PROBLEM DEFINE ---------------------------- */


#define BC_PROBLEM benchmark_ibm

                                
constexpr bool console_flush = false;

#define GPU_INDEX 0


/* --------------------------  SIMULATION DEFINES -------------------------- */
constexpr unsigned int N_GPUS = 1;    // Number of GPUS to use
constexpr unsigned int GPUS_TO_USE[N_GPUS] = {0};    // Which GPUs to use

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

#define CASE_DIRECTORY cases
#define CASE_CONSTANTS STR(CASE_DIRECTORY/BC_PROBLEM/constants.inc)
#define CASE_OUTPUTS STR(CASE_DIRECTORY/BC_PROBLEM/output.inc)
#define CASE_MODEL STR(CASE_DIRECTORY/BC_PROBLEM/model.inc)
#define CASE_BC_INIT STR(CASE_DIRECTORY/BC_PROBLEM/bc_initialization.inc)
#define CASE_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/bc_definition.inc)
#define CASE_FLOW_INITIALIZATION STR(CASE_DIRECTORY/BC_PROBLEM/flow_initialization.inc)
#define CASE_TREAT_DATA STR(CASE_DIRECTORY/BC_PROBLEM/treat_data.inc)
#define VOXEL_BC_DEFINE STR(../../CASE_DIRECTORY/voxel/bc_definition.inc)

#define COLREC_DIRECTORY colrec
#define COLREC_COLLISION STR(COLREC_DIRECTORY/COLLISION_TYPE/collision.inc)
#define COLREC_RECONSTRUCTION STR(COLREC_DIRECTORY/COLLISION_TYPE/reconstruction.inc)

#define CASE_G_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/g_bc_definition.inc)
#define COLREC_G_RECONSTRUCTION STR(COLREC_DIRECTORY/G_SCALAR/reconstruction.inc)
#define COLREC_G_COLLISION STR(COLREC_DIRECTORY/G_SCALAR/collision.inc)

#define CASE_AXX_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/axx_bc_definition.inc)
#define COLREC_AXX_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_xx.inc)
#define COLREC_AXX_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision.inc)

#define CASE_AXY_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/axy_bc_definition.inc)
#define COLREC_AXY_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_xy.inc)
#define COLREC_AXY_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision.inc)

#define CASE_AXZ_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/axz_bc_definition.inc)
#define COLREC_AXZ_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_xz.inc)
#define COLREC_AXZ_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision.inc)

#define CASE_AYY_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/ayy_bc_definition.inc)
#define COLREC_AYY_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_yy.inc)
#define COLREC_AYY_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision.inc)

#define CASE_AYZ_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/ayz_bc_definition.inc)
#define COLREC_AYZ_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_yz.inc)
#define COLREC_AYZ_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision.inc)

#define CASE_AZZ_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/azz_bc_definition.inc)
#define COLREC_AZZ_RECONSTRUCTION STR(COLREC_DIRECTORY/AIJ_SCALAR/reconstruction_zz.inc)
#define COLREC_AZZ_COLLISION STR(COLREC_DIRECTORY/AIJ_SCALAR/collision.inc)


#define CASE_PARTICLE_CREATE STR(../../CASE_DIRECTORY/BC_PROBLEM/particleCreation.inc)

// Some compiler timer functions and auxiliaty compute macros

#ifndef myMax
#define myMax(a,b)            (((a) > (b)) ? (a) : (b))
#endif //!myMax

#ifndef myMin
#define myMin(a,b)            (((a) < (b)) ? (a) : (b))
#endif //!myMin


constexpr dfloat constexprSqrt(dfloat x, dfloat curr, dfloat prev) {
    return (curr == prev) ? curr : constexprSqrt(x, 0.5 * (curr + x / curr), curr);
}

constexpr dfloat invSqrtNewton(dfloat x, dfloat curr, dfloat prev) {
    return (curr == prev) ? curr : invSqrtNewton(x, curr * (1.5 - 0.5 * x * curr * curr), curr);
}

constexpr dfloat sqrtt(dfloat x) {
    return (x >= 0 && x < std::numeric_limits<dfloat>::infinity())
        ? constexprSqrt(x, x, 0)
        : std::numeric_limits<dfloat>::quiet_NaN();
}

constexpr dfloat invSqrtt(dfloat x) {
    return (x > 0 && x < std::numeric_limits<dfloat>::infinity())
        ? invSqrtNewton(x, 1.0 / x, 0)
        : std::numeric_limits<dfloat>::quiet_NaN();
}


constexpr dfloat constexprLnHelper(dfloat y, int n, dfloat sum) {
    if (n > 10) {
        return sum;
    }
    dfloat term = y;
    for (int i = 0; i < n - 1; ++i) {
        term *= y * y;
    }
    return constexprLnHelper(y, n + 1, sum + term / (2.0 * n - 1.0));
}

constexpr dfloat constexprLn(dfloat x) {
    if (x <= 0.0) {
        return std::numeric_limits<dfloat>::quiet_NaN();
    }
    dfloat y = (x - 1.0) / (x + 1.0);
    return 2.0 * constexprLnHelper(y, 1, 0.0);
}

// Compile-time power of 2 checker
constexpr bool isPowerOfTwo(int x) {
    return (x & (x - 1)) == 0 && x != 0;
}

// Helper function to get the closest power of 2 under or equal to `n`
constexpr int closestPowerOfTwo(int n) {
    int power = 1;
    while (power * 2 <= n) {
        power *= 2;
    }
    return power;
}

struct BlockDim {
    int x, y, z;
};

// Compute optimal block dimensions
constexpr BlockDim findOptimalBlockDimensions(size_t maxElements) {
    int bestX = 1, bestY = 1, bestZ = 1;
    int closestVolume = 0;
    float bestForm = 0.0;    
    
    // Iterate over powers of 2 up to `maxElements` to find optimal dimensions
    for (int x = closestPowerOfTwo(maxElements); x >= 1; x /= 2) {
        for (int y = closestPowerOfTwo(maxElements / x); y >= 1; y /= 2) {
            for (int z = closestPowerOfTwo(maxElements / (x * y)); z >= 1; z /= 2) {
                if (x * y * z <= maxElements) {
                    int volume = x * y * z;
                    float form = 1.0/(1.0/x + 1.0/y + 1.0/z);
                    if (volume > closestVolume) {
                        bestX = x;
                        bestY = y;
                        bestZ = z;
                        closestVolume = volume;
                        bestForm = form;
                    } else if (volume == closestVolume && form > bestForm) {
                        bestX = x;
                        bestY = y;
                        bestZ = z;
                        bestForm = form;
                    }
                }
            }
        }
    }
    return {bestX, bestY, bestZ};
}

// swap 32-bit word
static inline uint32_t swap32(uint32_t v) {
    return  (v<<24) | 
           ((v<<8)&0x00FF0000) |
           ((v>>8)&0x0000FF00) |
            (v>>24);
}

// swap 64-bit word
static inline uint64_t swap64(uint64_t v) {
    return  (v<<56) |
           ((v<<40)&0x00FF000000000000ULL) |
           ((v<<24)&0x0000FF0000000000ULL) |
           ((v<<8 )&0x000000FF00000000ULL) |
           ((v>>8 )&0x00000000FF000000ULL) |
           ((v>>24)&0x0000000000FF0000ULL) |
           ((v>>40)&0x000000000000FF00ULL) |
            (v>>56);
}


#include CASE_CONSTANTS
#include CASE_MODEL
#include CASE_OUTPUTS


#include "definitions.h"
#include "nnf.h"


// --- compile-time checks ---
static_assert(NX >= BLOCK_NX, "NX must be >= BLOCK_NX, Update block size in definitions.h or increase domain in constants.inc");
static_assert(NY >= BLOCK_NY, "NY must be >= BLOCK_NY, Update block size in definitions.h or increase domain in constants.inc");
static_assert(NZ >= BLOCK_NZ, "NZ must be >= BLOCK_NZ, Update block size in definitions.h or increase domain in constants.inc");


#endif //__VAR_H