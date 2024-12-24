/**
*   @file globalStructs.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Global general structs
*   @version 0.3.0
*   @date 26/08/2020
*/

#ifndef __GLOBAL_STRUCTS_H
#define __GLOBAL_STRUCTS_H

#include "var.h"
#include "errorDef.h"

/*
*   Struct for dfloat in x, y, z
*/
typedef struct dfloat3 {
    dfloat x;
    dfloat y;
    dfloat z;

    __host__ __device__
    dfloat3(dfloat x = 0, dfloat y = 0, dfloat z = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    // Element-wise addition
    __host__ __device__
    friend dfloat3 operator+(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    // Element-wise subtraction
    __host__ __device__
    friend dfloat3 operator-(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    // Element-wise multiplication
    __host__ __device__
    friend dfloat3 operator*(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x * b.x, a.y * b.y, a.z * b.z);
    }

    // Element-wise division
    __host__ __device__
    friend dfloat3 operator/(const dfloat3& a, const dfloat3& b) {
        return dfloat3(a.x / b.x, a.y / b.y, a.z / b.z);
    }
    
    //between 1 dfloat and dfloat3
    // Element-wise addition with scalar
    __host__ __device__
    friend dfloat3 operator+(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
    }
    // Element-wise addition with scalar
    __host__ __device__
    friend dfloat3 operator+(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar + vec.x, scalar + vec.y, scalar + vec.z);
    }

    // Element-wise subtraction with scalar
    __host__ __device__
    friend dfloat3 operator-(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x - scalar, vec.y - scalar, vec.z - scalar);
    }
    // Element-wise subtraction with scalar
    __host__ __device__
    friend dfloat3 operator-(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar - vec.x, scalar - vec.y, scalar - vec.z);
    }

    // Element-wise multiplication with scalar
    __host__ __device__
    friend dfloat3 operator*(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x * scalar, vec.y * scalar, vec.z * scalar);
    }
    // Element-wise multiplication with scalar
    __host__ __device__
    friend dfloat3 operator*(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
    }

    // Element-wise division with scalar
    __host__ __device__
    friend dfloat3 operator/(const dfloat3& vec, const dfloat scalar) {
        return dfloat3(vec.x / scalar, vec.y / scalar, vec.z / scalar);
    }
    // Element-wise division with scalar
    __host__ __device__
    friend dfloat3 operator/(const dfloat scalar, const dfloat3& vec) {
        return dfloat3(scalar / vec.x, scalar / vec.y, scalar / vec.z);
    }


} dfloat3;

/*
*   Struct for dfloat in x, y, z, w (quartenion)
*/
typedef struct dfloat4{
    dfloat x;
    dfloat y;
    dfloat z;
    dfloat w;

    __host__ __device__
    dfloat4(dfloat x = 0, dfloat y = 0, dfloat z = 0, dfloat w = 0)
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }
} dfloat4;

typedef struct dfloat6{
    dfloat xx;
    dfloat yy;
    dfloat zz;
    dfloat xy;
    dfloat xz;
    dfloat yz;

    __host__ __device__
    dfloat6(dfloat xx = 0, dfloat yy = 0, dfloat zz = 0, dfloat xy = 0, dfloat xz = 0, dfloat yz = 0)
    {
        this->xx = xx;
        this->yy = yy;
        this->zz = zz;
        this->xy = xy;
        this->xz = xz;
        this->yz = yz;
    }
} dfloat6;


typedef struct ghostData {
    dfloat* X_0;
    dfloat* X_1;
    dfloat* Y_0;
    dfloat* Y_1;
    dfloat* Z_0;
    dfloat* Z_1;
} GhostData;


typedef struct ghostInterfaceData  {
    ghostData fGhost;
    ghostData gGhost;
    ghostData h_fGhost;

    #ifdef SECOND_DIST
        ghostData g_fGhost;
        ghostData g_gGhost;
        ghostData g_h_fGhost;
    #endif
    #ifdef A_XX_DIST
        ghostData Axx_fGhost;
        ghostData Axx_gGhost;
        ghostData Axx_h_fGhost;
    #endif 
    #ifdef A_XY_DIST
        ghostData Axy_fGhost;
        ghostData Axy_gGhost;
        ghostData Axy_h_fGhost;
    #endif 
    #ifdef A_XZ_DIST
        ghostData Axz_fGhost;
        ghostData Axz_gGhost;
        ghostData Axz_h_fGhost;
    #endif
    #ifdef A_YY_DIST
        ghostData Ayy_fGhost;
        ghostData Ayy_gGhost;
        ghostData Ayy_h_fGhost;
    #endif
    #ifdef A_YZ_DIST
        ghostData Ayz_fGhost;
        ghostData Ayz_gGhost;
        ghostData Ayz_h_fGhost;
    #endif 
    #ifdef A_ZZ_DIST
        ghostData Azz_fGhost;
        ghostData Azz_gGhost;
        ghostData Azz_h_fGhost;
    #endif

    #ifdef COMPUTE_VEL_DIVERGENT_FINITE_DIFFERENCE
        ghostData f_uGhost;
        ghostData g_uGhost;
        ghostData h_f_uGhost;
    #endif

    #ifdef COMPUTE_CONF_DIVERGENT_FINITE_DIFFERENCE
        ghostData conf_fGhost;
        ghostData conf_gGhost;
        ghostData conf_h_fGhost;
    #endif

} GhostInterfaceData;

#endif //__GLOBAL_STRUCTS_H
