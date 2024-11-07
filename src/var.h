#ifndef __VAR_H
#define __VAR_H


#include <builtin_types.h>  // for devices variables
#include <stdint.h>         // for uint32_t

#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

#define SINGLE_PRECISION 
#ifdef SINGLE_PRECISION
    typedef float dfloat;      // single precision
#endif
#ifdef DOUBLE_PRECISION
    typedef double dfloat;      // double precision
#endif



/* ----------------------------- PROBLEM DEFINE ---------------------------- */

#define BC_PROBLEM benchmark
                                
constexpr bool console_flush = false;



#define GPU_INDEX 0
/* --------------------------  SIMULATION DEFINES -------------------------- */

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

#define CASE_DIRECTORY cases
#define CASE_CONSTANTS STR(CASE_DIRECTORY/BC_PROBLEM/constants.inc)
#define CASE_OUTPUTS STR(CASE_DIRECTORY/BC_PROBLEM/output.inc)
#define CASE_MODEL STR(CASE_DIRECTORY/BC_PROBLEM/model.inc)
#define CASE_BC_INIT STR(CASE_DIRECTORY/BC_PROBLEM/bc_initialization.inc)
#define CASE_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/bc_definition.inc)
#define CASE_FLOW_INITIALIZATION STR(CASE_DIRECTORY/BC_PROBLEM/flow_initialization.inc)
#define VOXEL_BC_DEFINE STR(../../CASE_DIRECTORY/voxel/bc_definition.inc)

#define COLREC_DIRECTORY colrec
#define COLREC_COLLISION STR(COLREC_DIRECTORY/COLLISION_TYPE/collision.inc)
#define COLREC_RECONSTRUCTIONS STR(COLREC_DIRECTORY/COLLISION_TYPE/reconstruction.inc)

#define CASE_G_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/g_bc_definition.inc)
#define COLREC_G_RECONSTRUCTIONS STR(COLREC_DIRECTORY/G_SCALAR/reconstruction.inc)
#define COLREC_G_COLLISION STR(COLREC_DIRECTORY/G_SCALAR/collision.inc)


// Some compiler timer functions
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



#include CASE_MODEL
#include CASE_CONSTANTS
#include CASE_OUTPUTS

#include "definitions.h"

#endif //__VAR_H