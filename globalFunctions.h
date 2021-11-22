
#ifndef __GLOBAL_FUNCTIONS_H
#define __GLOBAL_FUNCTIONS_H

#include <builtin_types.h> // for device variables
#include "var.h"

__host__ __device__
    dfloat __forceinline__
    gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) ->
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}

__host__ __device__
    size_t __forceinline__
    idxMom(
        const int tx,
        const int ty,
        const int tz,
        const int mom,
        const int bx,
        const int by,
        const int bz)
{

    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (mom + NUMBER_MOMENTS * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz))))));
}

__device__ int __forceinline__
idxPopX(
    const int ty,
    const int tz,
    const int pop,
    const int bx,
    const int by,
    const int bz)
{

    /*idx //  D   pop //  D   pop
    D3Q19
    0   //  1   1   //  -1  2       
    1   //  1   7   //  -1  8
    3   //  1   9   //  -1  10
    4   //  1   13  //  -1  14
    5   //  1   15  //  -1  16
    D3Q27
    6   //  1   19  //  -1  20   
    7   //  1   21  //  -1  22
    8   //  1   23  //  -1  24
    9   //  1   26  //  -1  25
    */

    return ty + BLOCK_NY * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__device__ int __forceinline__
idxPopY(
    const int tx,
    const int tz,
    const int pop,
    const int bx,
    const int by,
    const int bz)
{

    /* 
    idx //  D   pop //  D   pop
    D3Q19
    0   //  1   3   //  -1  4       
    1   //  1   7   //  -1  8
    3   //  1   11  //  -1  12
    4   //  1   14  //  -1  13
    5   //  1   17  //  -1  18
    D3Q27
    6   //  1   19  //  -1  20   
    7   //  1   21  //  -1  22
    8   //  1   24  //  -1  23
    9   //  1   25  //  -1  26
    */
    return tx + BLOCK_NX * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__device__ int __forceinline__
idxPopZ(
    const int tx,
    const int ty,
    const int pop,
    const int bx,
    const int by,
    const int bz)
{

    /* 
    idx //  D   pop //  D   pop
    D3Q19
    0   //  1   5   //  -1  6       
    1   //  1   9   //  -1  10
    3   //  1   11  //  -1  12
    4   //  1   16  //  -1  15
    5   //  1   18  //  -1  17
    D3Q27
    6   //  1   19  //  -1  20   
    7   //  1   22  //  -1  21
    8   //  1   23  //  -1  24
    9   //  1   25  //  -1  26
    */

    return tx + BLOCK_NX * (ty + BLOCK_NY * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__host__ __device__
    size_t __forceinline__
    idxPopBlock(const unsigned int tx, const unsigned int ty, const unsigned int tz, const unsigned int pop)
{
    //return BLOCK_NX * (BLOCK_NY * (BLOCK_NZ * pop + tz) + ty) + tx;
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ *(pop)) );
}

__host__ __device__
    size_t __forceinline__
    idxScalarGlobal(unsigned int x, unsigned int y, unsigned int z)
{
    //return NX * (NY * z + y) + x;
    return x + NX * (y + NY*(z));
}

#endif // !__GLOBAL_FUNCTIONS_H
