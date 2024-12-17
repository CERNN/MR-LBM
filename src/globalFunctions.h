
#ifndef __GLOBAL_FUNCTIONS_H
#define __GLOBAL_FUNCTIONS_H

#include <builtin_types.h> // for device variables
#include "var.h"
#include "globalStructs.h"

__host__ __device__
    dfloat __forceinline__
    gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) ->
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}   
/*
* rhoVar 0
* uxVar  1
* uyVar  2
* uzVar  3
* pixx   4
* pixy   5
* pixz   6
* piyy   7
* piyz   8
* pizz   9
*/
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
    2   //  1   9   //  -1  10
    3   //  1   13  //  -1  14
    4   //  1   15  //  -1  16
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
    2   //  1   11  //  -1  12
    3   //  1   14  //  -1  13
    4   //  1   17  //  -1  18
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
    2   //  1   11  //  -1  12
    3   //  1   16  //  -1  15
    4   //  1   18  //  -1  17
    D3Q27
    6   //  1   19  //  -1  20   
    7   //  1   22  //  -1  21
    8   //  1   23  //  -1  24
    9   //  1   25  //  -1  26
    */

    return tx + BLOCK_NX * (ty + BLOCK_NY * (pop + QF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}
#if defined(SECOND_DIST) || defined(A_XX_DIST) || defined(A_XY_DIST) || defined(A_XZ_DIST) || defined(A_YY_DIST) || defined(A_YZ_DIST) || defined(A_ZZ_DIST)

__device__ int __forceinline__
g_idxPopX(
    const int ty,
    const int tz,
    const int pop,
    const int bx,
    const int by,
    const int bz)
{

    return ty + BLOCK_NY * (tz + BLOCK_NZ * (pop + GF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__device__ int __forceinline__
g_idxPopY(
    const int tx,
    const int tz,
    const int pop,
    const int bx,
    const int by,
    const int bz)
{
    return tx + BLOCK_NX * (tz + BLOCK_NZ * (pop + GF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__device__ int __forceinline__
g_idxPopZ(
    const int tx,
    const int ty,
    const int pop,
    const int bx,
    const int by,
    const int bz)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (pop + GF * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}
#endif

__host__ __device__
    size_t __forceinline__
    idxPopBlock(const unsigned int tx, const unsigned int ty, const unsigned int tz, const unsigned int pop)
{
    //return BLOCK_NX * (BLOCK_NY * (BLOCK_NZ * pop + tz) + ty) + tx;
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ *(pop)) );
}

__host__ __device__
    size_t __forceinline__
    idxScalarBlock(
        const int tx,
        const int ty,
        const int tz,
        const int bx,
        const int by,
        const int bz)
{

    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * (bz)))));
}

__host__ __device__
    size_t __forceinline__
    idxScalarGlobal(unsigned int x, unsigned int y, unsigned int z)
{
    //return NX * (NY * z + y) + x;
    return x + NX * (y + NY*(z));
}

#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
__host__ __device__ __forceinline__ 
size_t idxVelBlock(const unsigned int tx, const unsigned int ty, const unsigned int tz, const unsigned int uIndex)
{
    return (tx + HALO_SIZE) + (BLOCK_NX + 2 * HALO_SIZE) * ((ty + HALO_SIZE) + (BLOCK_NY + 2 * HALO_SIZE) * ((tz + HALO_SIZE) + (BLOCK_NZ + 2 * HALO_SIZE) * uIndex));
}
#endif


/**
*   @brief Compute the dot product of two vectors.
*   @param v1: First vector.
*   @param v2: Second vector.
*   @return The dot product of v1 and v2.
*/
__host__ __device__
dfloat dot_product(dfloat3 v1, dfloat3 v2);
/**
*   @brief Compute the cross product of two vectors.
*   @param v1: First vector.
*   @param v2: Second vector.
*   @return The cross product vector of v1 and v2.
*/
__host__ __device__
dfloat3 cross_product(dfloat3 v1, dfloat3 v2);

/**
*   @brief Determine the length of a vector
*   @param v: Vector to be computed.
*   @return The vector length.
*/
__host__ __device__
dfloat vector_length(dfloat3 v);

/**
*   @brief Normalize a vector.
*   @param v: Vector to be normalized.
*   @return The normalized vector.
*/
__host__ __device__
dfloat3 vector_normalize(dfloat3 v);
/**
*   @brief Compute the transpose of a 3x3 matrix.
*   @param matrix: The input 3x3 matrix to be transposed.
*   @param result: The output 3x3 matrix that will contain the transposed matrix.
*/
__host__ __device__
void transpose_matrix_3x3(dfloat matrix[3][3], dfloat result[3][3]);

/**
*   @brief Multiply two 3x3 matrices.
*   @param A: The first 3x3 matrix to be multiplied.
*   @param B: The second 3x3 matrix to be multiplied.
*   @param result: The output 3x3 matrix that will contain the product of matrices A and B..
*/
__host__ __device__
void multiply_matrices_3x3(dfloat A[3][3], dfloat B[3][3], dfloat result[3][3]);
/**
*   @brief Convert a dfloat6 structure to a 3x3 matrix.
*   @param I: dfloat6 structure containing inertia tensor components.
*   @param invA: Output 3x3 matrix.
*/
__host__ __device__
void dfloat6_to_matrix(dfloat6 I, dfloat M[3][3]);

/**
*   @brief Convert a 3x3 matrix to a dfloat6 structure.
*   @param M: Input 3x3 matrix.
*   @return I: dfloat6 structure to store the inertia tensor components.
*/
__host__ __device__
dfloat6 matrix_to_dfloat6(dfloat M[3][3]);
/**
*   @brief Compute the conjugate of a quaternion.
*   @param q: Quaternion to be conjugated.
*   @return The conjugate of q.
*/
__host__ __device__
dfloat4 quart_conjugate(dfloat4 q);
/**
*   @brief Convert a quaternion to a rotation matrix.
*   @param q: Quaternion to be converted.
*   @param R: Output rotation matrix.
*/
__host__ __device__
void quart_to_rotation_matrix(dfloat4 q, dfloat R[3][3]);
/**
*   @brief Rotate a vector by a rotation matrix.
*   @param v: Vector to be rotated.
*   @param R: Rotation matrix.
*   @return The rotated vector.
*/
__host__ __device__
dfloat3 rotate_vector_by_matrix(dfloat R[3][3],dfloat3 v);
/**
*   @brief Rotate a vector by a quaternion (using rotation matrix).
*   @param v: Vector to be rotated.
*   @param q: Quaternion representing rotation.
*   @return The rotated vector.
*/
__host__ __device__
dfloat3 rotate_vector_by_quart_R(dfloat3 v, dfloat4 q);
/**
*   @brief Compute the rotation quaternion that aligns two vectors.
*   @param v1: First vector.
*   @param v2: Second vector.
*   @return Quaternion representing the rotation from v1 to v2.
*/
__host__ __device__
dfloat4 compute_rotation_quart(dfloat3 v1, dfloat3 v2);
/**
*   @brief Convert an axis-angle representation to a quaternion.
*   @param axis: Rotation axis.
*   @param angle: Rotation angle.
*   @return The quaternion representation of the axis-angle rotation.
*/
__host__ __device__
dfloat4 axis_angle_to_quart(dfloat3 axis, dfloat angle);
/**
*   @brief Rotate a 3x3 matrix using a quaternion.
*   @param q: The quaternion representing the rotation.
*   @param I: The 3x3 matrix to be rotated.
*/
__host__ __device__
void rotate_matrix_by_R_w_quart(dfloat4 q, dfloat I[3][3]);
/**
*   @brief Rotate an inertia tensor represented as a 6-component structure using a quaternion.
*   @param q: The quaternion representing the rotation.
*   @param I6: The inertia tensor in the form of a 6-component structure.
*   @return The rotated inertia tensor as a 6-component structure.
*/
__host__ __device__
dfloat6 rotate_inertia_by_quart(dfloat4 q, dfloat6 I6);







#endif // !__GLOBAL_FUNCTIONS_H
