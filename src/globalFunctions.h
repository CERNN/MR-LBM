/**
 *  @file globalFunctions.h
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief Global functions
 *  @version 0.4.0
 *  @date 01/09/2025
 */



#ifndef __GLOBAL_FUNCTIONS_H
#define __GLOBAL_FUNCTIONS_H

#include <builtin_types.h> // for device variables
#include "var.h"
#include "globalStructs.h"
#ifdef PARTICLE_MODEL
#include "particles/models/ibm/ibmVar.h"
#endif

/**
 * @brief Compute the equilibrium distribution function
 * @param rhow: local density
 * @param uc3: 3 * (c_i . u)
 * @param p1_muu: 1 - 1.5 * (u . u)
 * @return f_eq
 */
__host__ __device__
    dfloat __forceinline__
    gpu_f_eq(const dfloat rhow, const dfloat uc3, const dfloat p1_muu)
{
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 + uc * uc * 4.5) ->
    // f_eq = rho_w * (1 - uu * 1.5 + uc * 3 * ( 1 + uc * 1.5)) ->
    return (rhow * (p1_muu + uc3 * (1.0 + uc3 * 0.5)));
}   

// ****************************************************************************
// ***************************   INDEX FUNCTIONS  *****************************
// ****************************************************************************





/**
 * @brief Compute the index for moment array
 * @param tx: thread x index
 * @param ty: thread y index
 * @param tz: thread z index
 * @param mom: moment index (0 to NUMBER_MOMENTS-1)
 * @param bx: block x index
 * @param by: block y index
 * @param bz: block z index
 * @return linear index for moment array
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

/**
 * @brief Compute the index for population array in x direction of the interface
 * @param ty: thread y index
 * @param tz: thread z index
 * @param pop: population index
 * @param bx: block x index
 * @param by: block y index
 * @param bz: block z index
 * @return linear index for population array
 */
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

/**
 * @brief Compute the index for population array in y direction of the interface
 * @param tx: thread x index
 * @param tz: thread z index
 * @param pop: population index
 * @param bx: block x index
 * @param by: block y index
 * @param bz: block z index
 * @return linear index for population array
 */
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

/**
 * @brief Compute the index for population array in z direction of the interface
 * @param tx: thread x index
 * @param ty: thread y index
 * @param pop: population index
 * @param bx: block x index
 * @param by: block y index
 * @param bz: block z index
 * @return linear index for population array
 */
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

#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
__device__ int __forceinline__
g_idxUX(
    const int ty,
    const int tz,
    const int dir,
    const int bx,
    const int by,
    const int bz)
{
    return dir + 3*(ty + BLOCK_NY*(tz + BLOCK_NZ*(bx + NUM_BLOCK_X*(by+NUM_BLOCK_Y*(bz)))));
    //return ty + BLOCK_NY * (tz + BLOCK_NZ * (dir + 3 * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__device__ int __forceinline__
g_idxUY(
    const int tx,
    const int tz,
    const int dir,
    const int bx,
    const int by,
    const int bz)
{
    return dir + 3*(tx + BLOCK_NX*(tz + BLOCK_NZ*(bx + NUM_BLOCK_X*(by+NUM_BLOCK_Y*(bz)))));
    //return tx + BLOCK_NX * (tz + BLOCK_NZ * (dir + 3 * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}

__device__ int __forceinline__
g_idxUZ(
    const int tx,
    const int ty,
    const int dir,
    const int bx,
    const int by,
    const int bz)
{
    return dir + 3*(tx + BLOCK_NX*(ty + BLOCK_NY*(bx + NUM_BLOCK_X*(by+NUM_BLOCK_Y*(bz)))));
    //return tx + BLOCK_NX * (ty + BLOCK_NY * (dir + 3 * (bx + NUM_BLOCK_X * (by + NUM_BLOCK_Y * bz))));
}
#endif

#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
__device__ int __forceinline__
g_idxConfX(
    const int ty,
    const int tz,
    const int dir,
    const int bx,
    const int by,
    const int bz)
{

    return dir + 6*(ty + BLOCK_NY*(tz + BLOCK_NZ*(bx + NUM_BLOCK_X*(by+NUM_BLOCK_Y*(bz)))));
}

__device__ int __forceinline__
g_idxConfY(
    const int tx,
    const int tz,
    const int dir,
    const int bx,
    const int by,
    const int bz)
{
    return dir + 6*(tx + BLOCK_NX*(tz + BLOCK_NZ*(bx + NUM_BLOCK_X*(by+NUM_BLOCK_Y*(bz)))));
}

__device__ int __forceinline__
g_idxConfZ(
    const int tx,
    const int ty,
    const int dir,
    const int bx,
    const int by,
    const int bz)
{
    return dir + 6*(tx + BLOCK_NX*(ty + BLOCK_NY*(bx + NUM_BLOCK_X*(by+NUM_BLOCK_Y*(bz)))));
}
#endif


/**
 * @brief Compute the index for population array in a block
 * @param tx: thread x index
 * @param ty: thread y index
 * @param tz: thread z index
 * @param pop: population index
 * @return linear index for population array in a block
 */
__host__ __device__
    size_t __forceinline__
    idxPopBlock(const unsigned int tx, const unsigned int ty, const unsigned int tz, const unsigned int pop)
{
    //return BLOCK_NX * (BLOCK_NY * (BLOCK_NZ * pop + tz) + ty) + tx;
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ *(pop)) );
}

/**
 * @brief Compute the index for scalar array in a block
 * @param tx: thread x index
 * @param ty: thread y index
 * @param tz: thread z index
 * @param bx: block x index
 * @param by: block y index
 * @param bz: block z index
 * @return linear index for scalar array
 */
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

/**
 * @brief Compute the index for scalar array in the global domain
 * @param x: x index
 * @param y: y index
 * @param z: z index
 * @return linear index for scalar array in the global domain
 */
__host__ __device__
    size_t __forceinline__
    idxScalarGlobal(unsigned int x, unsigned int y, unsigned int z)
{
    //return NX * (NY * z + y) + x;
    return x + NX * (y + NY*(z));
}


#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
//   @note: not unsigned because it uses negative values for thread index to pad from the halo
__host__ __device__ __forceinline__ 
size_t idxVelBlock(const int tx, const int ty, const int tz, const int uIndex)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ *(uIndex)) );
}
#endif


#ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
/**
 *  @brief Compute linear array index
 *  @param xx: 0
 *  @param xy: 1
 *  @param xz: 2
 *  @param yy: 3
 *  @param yz: 4
 *  @param zz: 5
 *  @note: not unsigned because it uses negative values for thread index to pad from the halo
 */
__host__ __device__ __forceinline__ 
size_t idxConfBlock(const int tx, const int ty, const int tz, const int confIndex)
{
    return tx + BLOCK_NX * (ty + BLOCK_NY * (tz + BLOCK_NZ *(confIndex)) );
}
#endif


// ****************************************************************************
// ************************   GENERAL OPERATIONS   ****************************
// ****************************************************************************

/**
 *  @brief Clamp a value to a given range.
 *  @param value: The value to be clamped.
 *  @return The clamped value.
 */
__host__ __device__
dfloat clamp01(dfloat value);
/**
 *  @brief Linearly interpolate between two vectors.
 *  @param v1: The start vector.
 *  @param v2: The end vector.
 *  @param t: Interpolation factor between 0 and 1.
 *  @return The interpolated vector between v1 and v2.
 */
__host__ __device__
dfloat3 vector_lerp(dfloat3 v1, dfloat3 v2, dfloat t);


// ****************************************************************************
// ************************   VECTOR OPERATIONS   *****************************
// ****************************************************************************

/**
 *  @brief Projects point P onto a plane defined by its normal vector and distance from the origin.
 *  @param P The point to project.
 *  @param n The normal vector of the plane.
 *  @param d The plane's distance from the origin along the normal vector.
 *  @return The projected point on the plane.
 */
__device__
dfloat3 planeProjection(dfloat3 P, dfloat3 n, dfloat d);
/**
 *  @brief Compute the dot product of two vectors.
 *  @param v1: First vector.
 *  @param v2: Second vector.
 *  @return The dot product of v1 and v2.
 */
__host__ __device__
dfloat dot_product(dfloat3 v1, dfloat3 v2);
/**
 *  @brief Compute the cross product of two vectors.
 *  @param v1: First vector.
 *  @param v2: Second vector.
 *  @return The cross product vector of v1 and v2.
 */
__host__ __device__
dfloat3 cross_product(dfloat3 v1, dfloat3 v2);

/**
 *  @brief Determine the length of a vector
 *  @param v: Vector to be computed.
 *  @return The vector length.
 */
__host__ __device__
dfloat vector_length(dfloat3 v);

/**
 *  @brief Normalize a vector.
 *  @param v: Vector to be normalized.
 *  @return The normalized vector.
 */
__host__ __device__
dfloat3 vector_normalize(dfloat3 v);

// ****************************************************************************
// ************************   MATRIX OPERATIONS   *****************************
// ****************************************************************************

/**
 *  @brief Compute the transpose of a 3x3 matrix.
 *  @param matrix: The input 3x3 matrix to be transposed.
 *  @param result: The output 3x3 matrix that will contain the transposed matrix.
 */
__host__ __device__
void transpose_matrix_3x3(dfloat matrix[3][3], dfloat result[3][3]);

/**
 *  @brief Multiply two 3x3 matrices.
 *  @param A: The first 3x3 matrix to be multiplied.
 *  @param B: The second 3x3 matrix to be multiplied.
 *  @param result: The output 3x3 matrix that will contain the product of matrices A and B..
 */
__host__ __device__
void multiply_matrices_3x3(dfloat A[3][3], dfloat B[3][3], dfloat result[3][3]);

/**
 *  @brief Result = scalar * A + B.
 *  @param scalar: the scalar that multiplies A before adding.
 *  @param A: The first 3x3 matrix to be multiplied.
 *  @param B: The second 3x3 matrix to be multiplied.
 *  @param result: The output 3x3 matrix that will contain the product of matrices A and B..
 */
__host__ __device__
void add_matrices_3x3(dfloat scalar, dfloat A[3][3], dfloat B[3][3], dfloat result[3][3]);

/**
 *  @brief Compute the determinant of a 3x3 matrix.
 *  @param A: The input 3x3 matrix whose determinant is to be computed.
 *  @return The determinant of the matrix A.
 */
__host__ __device__
dfloat determinant_3x3(dfloat A[3][3]);

/**
 *  @brief Compute the adjugate (or adjoint) of a 3x3 matrix.
 *  @param A: The input 3x3 matrix whose adjugate is to be computed.
 *  @param adj: The output 3x3 matrix that will contain the adjugate of matrix A.
 */
__host__ __device__
void adjugate_3x3(dfloat A[3][3], dfloat adj[3][3]);

/**
 *  @brief Compute the inverse of a 3x3 matrix.
 *  @param A: The input 3x3 matrix to be inverted.
 *  @param result: The output 3x3 matrix that will contain the inverse of matrix A.
 */
__host__ __device__
void inverse_3x3(dfloat A[3][3], dfloat result[3][3]);



// ****************************************************************************
// **********************   QUARTENION OPERATIONS   ***************************
// ****************************************************************************


/**
 *  @brief Compute the conjugate of a quaternion.
 *  @param q: Quaternion to be conjugated.
 *  @return The conjugate of q.
 */
__host__ __device__
dfloat4 quart_conjugate(dfloat4 q);

/**
 *  @brief Perform quaternion multiplication.
 *  @param q1: First quaternion.
 *  @param q2: Second quaternion.
 *  @return The product of q1 and q2.
 */
__host__ __device__
dfloat4 quart_multiplication(dfloat4 q1, dfloat4 q2);

// ****************************************************************************
// **********************   CONVERSION OPERATIONS   ***************************
// ****************************************************************************


/**
 *  @brief Convert a dfloat6 structure to a 3x3 matrix.
 *  @param I: dfloat6 structure containing inertia tensor components.
 *  @param invA: Output 3x3 matrix.
 */
__host__ __device__
void dfloat6_to_matrix(dfloat6 I, dfloat M[3][3]);

/**
 *  @brief Convert a 3x3 matrix to a dfloat6 structure.
 *  @param M: Input 3x3 matrix.
 *  @return I: dfloat6 structure to store the inertia tensor components.
 */
__host__ __device__
dfloat6 matrix_to_dfloat6(dfloat M[3][3]);

/**
 *  @brief Convert a quaternion to a rotation matrix.
 *  @param q: Quaternion to be converted.
 *  @param R: Output rotation matrix.
 */
__host__ __device__
void quart_to_rotation_matrix(dfloat4 q, dfloat R[3][3]);


/**
 *  @brief Convert Euler angles to a quaternion.
 *  @param roll: Rotation angle around x-axis.
 *  @param pitch: Rotation angle around y-axis.
 *  @param yaw: Rotation angle around z-axis.
 *  @return The quaternion representation of the Euler angles.
 *  @source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
 */
__host__ __device__
dfloat4 euler_to_quart(dfloat roll, dfloat pitch, dfloat yaw);

/**
 *  @brief Convert a quaternion to Euler angles.
 *  @param q: Quaternion to be converted.
 *  @return Euler angles representing the rotation of q.
 */
__host__ __device__
dfloat3 quart_to_euler(dfloat4 q);

/**
 *  @brief Computes the rotation matrix contructed from vector v1 with vector v2.
 *  @param v1 The initial vector.
 *  @param v2 The target vector to align with.
 *  @param R Output 3x3 rotation matrix.
 */
__device__
void rotationMatrixFromVectors(dfloat3 v1, dfloat3 v2, dfloat R[3][3]);
/**
 *  @brief Computes the rotation matrix that aligns three orthogonal vectors v1, v2, and v3.
 *  @param v1 The first vector (must be orthogonal to v2 and v3).
 *  @param v2 The second vector (must be orthogonal to v1 and v3).
 *  @param v3 The third vector (must be orthogonal to v1 and v2).
 *  @param R Output 3x3 rotation matrix.
 */
__device__
void rotationMatrixFromVectors(dfloat3 v1, dfloat3 v2, dfloat3 v3, dfloat R[3][3]);



// ****************************************************************************
// ***********************   ROTATION OPERATIONS   ****************************
// ****************************************************************************


/**
 *  @brief Rotate a vector by a rotation matrix.
 *  @param v: Vector to be rotated.
 *  @param R: Rotation matrix.
 *  @return The rotated vector.
 */
__host__ __device__
dfloat3 rotate_vector_by_matrix(dfloat R[3][3],dfloat3 v);

/**
 *  @brief Rotate a vector by a quaternion (using rotation matrix).
 *  @param v: Vector to be rotated.
 *  @param q: Quaternion representing rotation.
 *  @return The rotated vector.
 */
__host__ __device__
dfloat3 rotate_vector_by_quart_R(dfloat3 v, dfloat4 q);

/**
 *  @brief Compute the rotation quaternion that aligns two vectors.
 *  @param v1: First vector.
 *  @param v2: Second vector.
 *  @return Quaternion representing the rotation from v1 to v2.
 */
__host__ __device__
dfloat4 compute_rotation_quart(dfloat3 v1, dfloat3 v2);

/**
 *  @brief Convert an axis-angle representation to a quaternion.
 *  @param axis: Rotation axis.
 *  @param angle: Rotation angle.
 *  @return The quaternion representation of the axis-angle rotation.
 */
__host__ __device__
dfloat4 axis_angle_to_quart(dfloat3 axis, dfloat angle);

/**
 *  @brief Rotate a 3x3 matrix using a quaternion.
 *  @param q: The quaternion representing the rotation.
 *  @param I: The 3x3 matrix to be rotated.
 */
__host__ __device__
void rotate_matrix_by_R_w_quart(dfloat4 q, dfloat I[3][3]);

/**
 *  @brief Rotate an inertia tensor represented as a 6-component structure using a quaternion.
 *  @param q: The quaternion representing the rotation.
 *  @param I6: The inertia tensor in the form of a 6-component structure.
 *  @return The rotated inertia tensor as a 6-component structure.
 */
__host__ __device__
dfloat6 rotate_inertia_by_quart(dfloat4 q, dfloat6 I6);





#endif // !__GLOBAL_FUNCTIONS_H
