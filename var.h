#ifndef __VAR_H
#define __VAR_H


#include <builtin_types.h>  // for devices variables
#include <stdint.h>         // for uint32_t

#define _USE_MATH_DEFINES
#include <math.h>

#define SINGLE_PRECISION 
#ifdef SINGLE_PRECISION
    typedef float dfloat;      // single precision
#endif
#ifdef DOUBLE_PRECISION
    typedef double dfloat;      // double precision
#endif

#define D3Q19

/* ----------------------------- OUTPUT DEFINES ---------------------------- */

#define ID_SIM "000"            // prefix for simulation's files
#define PATH_FILES "TEST"  // path to save simulation's files


/* --------------------------  SIMULATION DEFINES -------------------------- */

constexpr int N_STEPS = 200;
#define MACR_SAVE (1)

constexpr int SCALE = 1;
constexpr int N = 64 * SCALE;
constexpr int NX = N * SCALE;        // size x of the grid 
                                    // (32 multiple for better performance)
constexpr int NY = N * SCALE;        // size y of the grid
constexpr int NZ = N * SCALE;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat TAU = 0.6;     // relaxation time
constexpr dfloat OMEGA = 1.0 / TAU;        // (tau)^-1
constexpr dfloat T_OMEGA = 1.0 -OMEGA;
constexpr dfloat TT_OMEGA = 1.0 -0.5*OMEGA;

constexpr dfloat RHO_0 = 1;         // initial rho

constexpr dfloat FX = 0.0;        // force in x
constexpr dfloat FY = 0.0;        // force in y
constexpr dfloat FZ = 0.0;        // force in z (flow direction in most cases)

#define SQRT_2 (1.41421356237309504880168872420969807856967187537)

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

constexpr dfloat inv_cs2 = 3.0;
constexpr dfloat cs2 = 1.0/inv_cs2;

// populations velocities vector
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };
#endif //D3Q19

#ifdef D3Q27
constexpr unsigned char Q = 27;         // number of velocities
constexpr unsigned char QF = 9*2;         // number of velocities on each face
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


constexpr dfloat inv_cs2 = 3.0;
constexpr dfloat cs2 = 1.0/inv_cs2;

// populations velocities vector
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1};
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1};
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1};

#endif //D3Q27


/* ------------------------------ MEMORY SIZE ------------------------------ */
const size_t BLOCK_NX = 8;
const size_t BLOCK_NY = 8;
const size_t BLOCK_NZ = 8;
const size_t BLOCK_LBM_SIZE = BLOCK_NX * BLOCK_NY * BLOCK_NZ;

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

#define MOMENT_ORDER 3
const size_t NUMBER_MOMENTS = (MOMENT_ORDER)* (MOMENT_ORDER + 1)* (MOMENT_ORDER + 2) / 6;

const size_t MEM_SIZE_BLOCK_LBM = sizeof(dfloat) * BLOCK_LBM_SIZE * NUMBER_MOMENTS;
const size_t MEM_SIZE_BLOCK_GHOST = sizeof(dfloat) * BLOCK_GHOST_SIZE * Q;
const size_t MEM_SIZE_BLOCK_TOTAL = MEM_SIZE_BLOCK_GHOST + MEM_SIZE_BLOCK_LBM;

const size_t NUMBER_LBM_POP_NODES = NX * NY * NZ;


//memory size
const size_t MEM_SIZE_SCALAR = sizeof(dfloat) * NUMBER_LBM_NODES;
const size_t MEM_SIZE_POP = sizeof(dfloat) * NUMBER_LBM_POP_NODES * Q;
const size_t MEM_SIZE_MOM = sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS;


/* --------------------------- AUXILIARY DEFINES --------------------------- */
#define IN_HOST 1       // variable accessible only for host
#define IN_VIRTUAL 2    // variable accessible for device and host

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);

#define SQRT_2 (1.41421356237309504880168872420969807856967187537)



#endif //__VAR_H