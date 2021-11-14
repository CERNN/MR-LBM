
#ifndef __GLOBAL_FUNCTIONS_H
#define __GLOBAL_FUNCTIONS_H


#include <builtin_types.h> // for device variables
#include "var.h"

__host__ __device__
dfloat __forceinline__ gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) -> 
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}




__host__ __device__
size_t __forceinline__ idxScalarGlobal(unsigned int x, unsigned int y, unsigned int z)
{
    return NX * ((size_t)NY * z + y) + x;
}

__host__ __device__ 
size_t __forceinline__ idxBlock(unsigned int bx, unsigned int by, unsigned int bz)
{
    return  (bx + (NUM_BLOCK_X)*by + (NUM_BLOCK_X * NUM_BLOCK_Y) * bz);
}

__host__ __device__
size_t __forceinline__ idxScalarBlock(unsigned int x, unsigned int y, unsigned int z)
{
    return BLOCK_NX * ((size_t)BLOCK_NY * z + y) + x;
}


__host__ __device__
size_t __forceinline__ idxPopBlock(const unsigned int x, const unsigned int y, const unsigned int z, const unsigned int pop)
{
    return BLOCK_NX * (BLOCK_NY * ((size_t)(BLOCK_NZ) * pop + z) + y) + x;
}

__host__ __device__
size_t __forceinline__ idxPopX( const unsigned int ty, const unsigned int tz, const unsigned int pop, const unsigned int bx, const unsigned int by, const unsigned int bz)
{
    /*return  ty + 
            tz * BLOCK_NY + 
            pop* BLOCK_NY * BLOCK_NZ + 
            bx * BLOCK_NY * BLOCK_NZ * QF + 
            by * BLOCK_NY * BLOCK_NZ * QF * NUM_BLOCK_X +
            bz * BLOCK_NY * BLOCK_NZ * QF * NUM_BLOCK_X * NUM_BLOCK_Y;*/
    /* 
    idx //  D   pop //  D   pop
    D3Q19
    0   //  1   1   
    1   //  1   7   
    2   //  1   9   
    3   //  1   13  
    4   //  1   15  
    5   //  -1  2       
    6   //  -1  8
    7   //  -1  10
    8   //  -1  14
    9   //  -1  16
    D3Q27
    6   //  1   19  //  -1  20   
    7   //  1   21  //  -1  22
    8   //  1   23  //  -1  24
    9   //  1   26  //  -1  25
    */
    return  ty + BLOCK_NY * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__host__ __device__
size_t __forceinline__ idxPopY( const unsigned int tx, const unsigned int tz, const unsigned int pop, const unsigned int bx, const unsigned int by, const unsigned int bz)
{
    /* 
    idx //  D   pop //  D   pop
    D3Q19
    0   //  1   3   
    1   //  1   7   
    2   //  1   11  
    3   //  1   14  
    4   //  1   17
    5   //  -1  4       
    6   //  -1  8
    7   //  -1  12
    8   //  -1  13
    9   //  -1  18  
    D3Q27
    5   //  1   19  //  -1  20   
    6   //  1   21  //  -1  22
    7   //  1   24  //  -1  23
    8   //  1   25  //  -1  26
    */
    return  tx + BLOCK_NX * (tz + BLOCK_NZ * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__host__ __device__
size_t __forceinline__ idxPopZ( const unsigned int tx, const unsigned int ty, const unsigned int pop, const unsigned int bx, const unsigned int by, const unsigned int bz)
{
    /* 
    idx //  D   pop //  D   pop
    D3Q19
    0   //  1   5   
    1   //  1   9   
    2   //  1   11  
    3   //  1   16  
    4   //  1   18
    5   //  -1  6       
    6   //  -1  10
    7   //  -1  12
    8   //  -1  15
    9   //  -1  17  
    D3Q27
    5   //  1   19  //  -1  20   
    6   //  1   22  //  -1  21
    7   //  1   23  //  -1  24
    8   //  1   25  //  -1  26
    */
    return  tx + BLOCK_NX * (ty + BLOCK_NY * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y  *bz))));
}








#endif // !__GLOBAL_FUNCTIONS_H
