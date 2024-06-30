#ifndef __DEFINITIONS_H
#define __DEFINITIONS_H

#include "var.h"


#if defined(POWERLAW) || defined(BINGHAM) || defined(BI_VISCOSITY)
    #define NON_NEWTONIAN_FLUID
#endif

constexpr dfloat OMEGA = 1.0 / TAU;        // (tau)^-1
constexpr dfloat OMEGAd2 = OMEGA/2.0; //OMEGA/2
constexpr dfloat OMEGAd9 = OMEGA/9.0;  //OMEGA/9
constexpr dfloat T_OMEGA = 1.0 - OMEGA; //1-OMEGA
constexpr dfloat TT_OMEGA = 1.0 - 0.5*OMEGA; //1.0 - OMEGA/2 
constexpr dfloat OMEGA_P1 = 1.0 + OMEGA; // 1+ OMEGA
constexpr dfloat TT_OMEGA_T3 = TT_OMEGA*3.0; //3*(1-0.5*OMEGA)


#define SQRT_2 (1.41421356237309504880168872420969807856967187537)

/* ------------------------------ PROBE DEFINES ------------------------------ */

constexpr int probe_x = (NX/2);
constexpr int probe_y = (NY/2);
constexpr int probe_z = (NZ_TOTAL/2);
constexpr int probe_index = probe_x + NX * (probe_y + NY*(probe_z));


/* ------------------------------ VELOCITY SET ------------------------------ */
#ifdef D3Q19
constexpr unsigned char Q = 19;        // number of velocities
constexpr unsigned char QF = 5;         // number of velocities on each face
constexpr dfloat W0 = 1.0 / 3;         // population 0 weight (0, 0, 0)
constexpr dfloat W1 = 1.0 / 18;        // adjacent populations (1, 0, 0)
constexpr dfloat W2 = 1.0 / 36;        // diagonal populations (1, 1, 0)

// velocities weight vector
__device__ const dfloat w[Q] = { W0,
    W1, W1, W1, W1, W1, W1,
    W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2
};

constexpr dfloat as2 = 3.0;
constexpr dfloat cs2 = 1.0/as2;

// populations velocities vector 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18  
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };

constexpr dfloat F_M_0_SCALE = 1.0;
constexpr dfloat F_M_I_SCALE = as2;
constexpr dfloat F_M_II_SCALE = as2*as2/2;
constexpr dfloat F_M_IJ_SCALE = as2*as2;

#endif //D3Q19

#ifdef D3Q27
constexpr unsigned char Q = 27;         // number of velocities
constexpr unsigned char QF = 9;         // number of velocities on each face
constexpr dfloat W0 = 8.0 / 27;        // weight dist 0 population (0, 0, 0)
constexpr dfloat W1 = 2.0 / 27;        // weight dist 1 populations (1, 0, 0)
constexpr dfloat W2 = 1.0 / 54;        // weight dist 2 populations (1, 1, 0)
constexpr dfloat W3 = 1.0 / 216;       // weight dist 3 populations (1, 1, 1)

// velocities weight vector
__device__ const dfloat w[Q] = { W0,
    W1, W1, W1, W1, W1, W1,
    W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2,
    W3, W3, W3, W3, W3, W3, W3, W3
};


constexpr dfloat as2 = 3.0;
constexpr dfloat cs2 = 1.0/as2;

// populations velocities vector 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1};
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1};
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1};

#endif //D3Q27


// #define SECOND_DIST

#ifdef D3G7

    constexpr unsigned char GQ = 7;        // number of velocities
    constexpr unsigned char GF = 1;         // number of velocities on each face
    constexpr dfloat gW0 = 1.0 / 4.0;         // population 0 weight (0, 0, 0)
    constexpr dfloat gW1 = 1.0 / 8.0;        // adjacent populations (1, 0, 0)
    // velocities weight vector
    __device__ const dfloat gw[GQ] = { 
        gW0,
        gW1, gW1, gW1, gW1, gW1, gW1};

    constexpr dfloat g_as2 = 4.0;
    constexpr dfloat g_cs2 = 1.0/g_as2;

    // populations velocities vector
    __device__ const char gcx[GQ] = { 0, 1,-1, 0, 0, 0, 0};
    __device__ const char gcy[GQ] = { 0, 0, 0, 1,-1, 0, 0};
    __device__ const char gcz[GQ] = { 0, 0, 0, 0, 0, 1,-1};
#endif

#ifdef D3G19

    constexpr unsigned char GQ = 19;        // number of velocities
    constexpr unsigned char GF = 5;         // number of velocities on each face
    constexpr dfloat gW0 = 1.0 / 3;         // population 0 weight (0, 0, 0)
    constexpr dfloat gW1 = 1.0 / 18;        // adjacent populations (1, 0, 0)
    constexpr dfloat gW2 = 1.0 / 36;        // diagonal populations (1, 1, 0)
    // velocities weight vector
    __device__ const dfloat gw[GQ] = { 
        gW0,
        gW1, gW1, gW1, gW1, gW1, gW1, 
        gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2
    };

    constexpr dfloat g_as2 = 3.0;
    constexpr dfloat g_cs2 = 1.0/g_as2;

    // populations velocities vector
    __device__ const char gcx[GQ] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
    __device__ const char gcy[GQ] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
    __device__ const char gcz[GQ] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };
#endif




constexpr dfloat ONESIXTH = 1.0/6.0;
constexpr dfloat ONETHIRD = 1.0/3.0;
/* ------------------------------ LES MODEL ------------------------------ */

#ifdef MODEL_CONST_SMAGORINSKY
constexpr dfloat CONST_SMAGORINSKY = 0.1;
constexpr dfloat INIT_VISC_TURB = 0.0;


constexpr dfloat Implicit_const = 2.0*SQRT_2*3*3/(RHO_0)*CONST_SMAGORINSKY*CONST_SMAGORINSKY;

#endif


/* ------------------------------ MEMORY SIZE ------------------------------ */
#include  "arrayIndex.h"


#define BLOCK_NX 8
#define BLOCK_NY 8
#ifdef SINGLE_PRECISION //some easy fix so doesnt forget to change size when changing float size
    #define BLOCK_NZ 4
#endif
#ifdef DOUBLE_PRECISION
    #define BLOCK_NZ 4
#endif

#define BLOCK_LBM_SIZE (BLOCK_NX * BLOCK_NY * BLOCK_NZ)

const size_t BLOCK_FACE_XY = BLOCK_NX * BLOCK_NY;
const size_t BLOCK_FACE_XZ = BLOCK_NX * BLOCK_NZ;
const size_t BLOCK_FACE_YZ = BLOCK_NY * BLOCK_NZ;
const size_t BLOCK_GHOST_SIZE = BLOCK_FACE_XY + BLOCK_FACE_XZ + BLOCK_FACE_YZ;

const size_t BLOCK_SIZE = BLOCK_LBM_SIZE + BLOCK_GHOST_SIZE;
//const size_t BLOCK_SIZE = (BLOCK_NX + 1) * (BLOCK_NY + 1) * (BLOCK_NZ + 1);

const size_t NUM_BLOCK_X = NX / BLOCK_NX;
const size_t NUM_BLOCK_Y = NY / BLOCK_NY;
const size_t NUM_BLOCK_Z = NZ / BLOCK_NZ;
const size_t NUM_BLOCK = NUM_BLOCK_X * NUM_BLOCK_Y * NUM_BLOCK_Z;

const size_t NUMBER_LBM_NODES = NUM_BLOCK * BLOCK_LBM_SIZE;
const size_t NUMBER_GHOST_FACE_XY = BLOCK_NX*BLOCK_NY*NUM_BLOCK_X*NUM_BLOCK_Y*NUM_BLOCK_Z;
const size_t NUMBER_GHOST_FACE_XZ = BLOCK_NX*BLOCK_NZ*NUM_BLOCK_X*NUM_BLOCK_Y*NUM_BLOCK_Z;
const size_t NUMBER_GHOST_FACE_YZ = BLOCK_NY*BLOCK_NZ*NUM_BLOCK_X*NUM_BLOCK_Y*NUM_BLOCK_Z;

//#define MOMENT_ORDER (1+2)
//#ifdef NON_NEWTONIAN_FLUID
//const size_t NUMBER_MOMENTS = (MOMENT_ORDER)* (MOMENT_ORDER + 1)* (MOMENT_ORDER + 2) / 6 + 1;
//#endif
//#ifndef NON_NEWTONIAN_FLUID
//const size_t NUMBER_MOMENTS = (MOMENT_ORDER)* (MOMENT_ORDER + 1)* (MOMENT_ORDER + 2) / 6;
//#endif




const size_t MEM_SIZE_BLOCK_LBM = sizeof(dfloat) * BLOCK_LBM_SIZE * NUMBER_MOMENTS;
const size_t MEM_SIZE_BLOCK_GHOST = sizeof(dfloat) * BLOCK_GHOST_SIZE * Q;
const size_t MEM_SIZE_BLOCK_TOTAL = MEM_SIZE_BLOCK_GHOST + MEM_SIZE_BLOCK_LBM;

const size_t NUMBER_LBM_POP_NODES = NX * NY * NZ;


//memory size
const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
const size_t MEM_SIZE_POP = sizeof(dfloat) * NUMBER_LBM_POP_NODES * Q;
const size_t MEM_SIZE_MOM = sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS;

/* ------------------------------ GPU DEFINES ------------------------------ */
const int N_THREADS = (NX%64?((NX%32||(NX<32))?NX:32):64); // NX or 32 or 64 
                                    // multiple of 32 for better performance.
const int CURAND_SEED = 0;          // seed for random numbers for CUDA
constexpr float CURAND_STD_DEV = 0.5; // standard deviation for random numbers 
                                    // in normal distribution
/* ------------------------------------------------------------------------- */


/* --------------------------- AUXILIARY DEFINES --------------------------- */
#define IN_HOST 1       // variable accessible only for host
#define IN_VIRTUAL 2    // variable accessible for device and host

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);

#define SQRT_2 (1.41421356237309504880168872420969807856967187537)


#ifndef myMax
#define myMax(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef myMin
#define myMin(a,b)            (((a) < (b)) ? (a) : (b))
#endif


#if defined(HO_RR) || defined(HOME_LBM)
    #define HIGH_ORDER_COLLISION
#endif

//#define DYNAMIC_SHARED_MEMORY
#ifdef DYNAMIC_SHARED_MEMORY
#define SHARED_MEMORY_SIZE 101376 //
#endif


#endif
