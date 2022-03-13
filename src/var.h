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

/* ----------------------------- BC DEFINES ---------------------------- */

#define BC_PROBLEM lidDrivenCavity

//#define BC_POPULATION_BASED
#define BC_MOMENT_BASED

/* ----------------------------- OUTPUT DEFINES ---------------------------- */

#define ID_SIM "001"            // prefix for simulation's files
#define PATH_FILES "TEST"  // path to save simulation's files
#define RANDOM_NUMBERS false    // to generate random numbers 
                                // (useful for turbulence)



#define GPU_INDEX 0
/* --------------------------  SIMULATION DEFINES -------------------------- */

constexpr int SCALE = 1;

#define MACR_SAVE (1)


constexpr int N = 128 * SCALE;
constexpr int NX = N;        // size x of the grid 
                                    // (32 multiple for better performance)
constexpr int NY = N;        // size y of the grid
constexpr int NZ = N;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat U_MAX = 0.10;  
constexpr dfloat RE = 100.0;	
constexpr dfloat L = N;
constexpr dfloat VISC = L*U_MAX / RE;


constexpr int N_STEPS = 10;


constexpr dfloat TAU = 0.5 + 3.0*VISC;     // relaxation time
constexpr dfloat OMEGA = 1.0 / TAU;        // (tau)^-1
constexpr dfloat OMEGAd2 = OMEGA/2.0;
constexpr dfloat OMEGAd3 = OMEGA/3.0;
constexpr dfloat OMEGAd9 = OMEGA/9.0; 
constexpr dfloat T_OMEGA = 1.0 - OMEGA;
constexpr dfloat TT_OMEGA = 1.0 - 0.5*OMEGA;
constexpr dfloat OMEGA_P1 = 1.0 + OMEGA;
constexpr dfloat TT_OMEGA_T3 = TT_OMEGA*3.0;

constexpr dfloat RHO_0 = 1.0;         // initial rho

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

constexpr dfloat as2 = 3.0;
constexpr dfloat cs2 = 1.0/as2;

// populations velocities vector 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18  
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };
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

constexpr dfloat ONESIXTH = 1.0/6.0;
constexpr dfloat ONETHIRD = 1.0/3.0;

/* ------------------------------ MEMORY SIZE ------------------------------ */
const size_t BLOCK_NX = 8;
const size_t BLOCK_NY = 8;
const size_t BLOCK_NZ = 4;
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




#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

#ifdef BC_POPULATION_BASED
    #define BC_DIRECTORY BoundaryConditions/IncludeMlbmBc_POP

    #define BC_PATH STR(BC_DIRECTORY/BC_PROBLEM)
#endif

#ifdef BC_MOMENT_BASED
    #define BC_DIRECTORY BoundaryConditions/IncludeMlbmBc_MOM

    #define BC_PATH STR(BC_DIRECTORY/BC_PROBLEM)
#endif

#define BC_DIRECTORY_INIT BoundaryConditions/Boundary_initialization_files
#define BC_INIT_PATH STR(BC_DIRECTORY_INIT/BC_PROBLEM)

#endif //__VAR_H