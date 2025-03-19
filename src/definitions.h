#ifndef __DEFINITIONS_H
#define __DEFINITIONS_H

#include "var.h"

/* --------------------------- CONSTANTS --------------------------- */

constexpr dfloat OMEGA = 1.0 / TAU;        // (tau)^-1
constexpr dfloat OMEGAd2 = OMEGA/2.0; //OMEGA/2
constexpr dfloat OMEGAd9 = OMEGA/9.0;  //OMEGA/9
constexpr dfloat T_OMEGA = 1.0 - OMEGA; //1-OMEGA
constexpr dfloat TT_OMEGA = 1.0 - 0.5*OMEGA; //1.0 - OMEGA/2 
constexpr dfloat OMEGA_P1 = 1.0 + OMEGA; // 1+ OMEGA
constexpr dfloat TT_OMEGA_T3 = TT_OMEGA*3.0; //3*(1-0.5*OMEGA)

#define SQRT_2  (1.41421356237309504880168872420969807856967187537)
#define SQRT_10 (3.162277660168379331998893544432718533719555139325)


constexpr dfloat ONESIXTH = 1.0/6.0;
constexpr dfloat ONETHIRD = 1.0/3.0;

/* --------------------------- AUXILIARY DEFINES --------------------------- */
#define IN_HOST 1       // variable accessible only for host
#define IN_VIRTUAL 2    // variable accessible for device and host

constexpr size_t BYTES_PER_GB = (1 << 30);
constexpr size_t BYTES_PER_MB = (1 << 20);

/* ------------------------------ VELOCITY SET ------------------------------ */
#ifdef D3Q19
    #include "includeFiles/velocitySets/D3Q19.inc"
#endif //D3Q19
#ifdef D3Q27
    #include "includeFiles/velocitySets/D3Q27.inc"
#endif //D3Q27

// #define SECOND_DIST
#ifdef D3G7
    #include "includeFiles/velocitySets/D3G7.inc"
#endif
#ifdef D3G19
    #include "includeFiles/velocitySets/D3G19.inc"
#endif

/* ------------------------------ MODEL MACROS ------------------------------ */


#if defined(POWERLAW) || defined(BINGHAM) || defined(BI_VISCOSITY)
    #define OMEGA_FIELD
#endif


#ifdef MODEL_CONST_SMAGORINSKY
constexpr dfloat CONST_SMAGORINSKY = 0.1;
constexpr dfloat INIT_VISC_TURB = 0.0;
constexpr dfloat Implicit_const = 2.0*SQRT_2*3*3/(RHO_0)*CONST_SMAGORINSKY*CONST_SMAGORINSKY;
#endif


/* ------------------------------ MEMORY SIZE ------------------------------ */
#include  "arrayIndex.h"

// Calculate maximum number of elements in a block
//#define DYNAMIC_SHARED_MEMORY
#ifdef DYNAMIC_SHARED_MEMORY
    #if (defined(SM_90) || defined(SM_100) || defined(SM_120))
        constexpr size_t SHARED_MEMORY_SIZE = 232448;  // sm90
    #endif
    #if (defined(SM_80)) || (defined(SM_87))
        constexpr size_t SHARED_MEMORY_SIZE = 166912;  // sm80
    #endif
    #if (defined(SM_86) || defined(SM_89))
        constexpr size_t SHARED_MEMORY_SIZE = 101376;  // sm86
    #endif
#endif

constexpr int SHARED_MEMORY_ELEMENT_SIZE = sizeof(dfloat) * (Q - 1);
#ifdef SHARED_MEMORY_LIMIT
    constexpr size_t MAX_ELEMENTS_IN_BLOCK = SHARED_MEMORY_LIMIT / SHARED_MEMORY_ELEMENT_SIZE;
#else
    constexpr int MAX_ELEMENTS_IN_BLOCK = 48128 / SHARED_MEMORY_ELEMENT_SIZE;
#endif

constexpr BlockDim optimalBlockDimArray = findOptimalBlockDimensions(MAX_ELEMENTS_IN_BLOCK);

//TODO: fix, is giving incopatibility issues with parallel reduction
#define BLOCK_NX 8
#define BLOCK_NY 8
#define BLOCK_NZ 8


#define BLOCK_LBM_SIZE (BLOCK_NX * BLOCK_NY * BLOCK_NZ)

const size_t BLOCK_LBM_SIZE_POP = BLOCK_LBM_SIZE * (Q - 1);

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
//#ifdef OMEGA_FIELD
//const size_t NUMBER_MOMENTS = (MOMENT_ORDER)* (MOMENT_ORDER + 1)* (MOMENT_ORDER + 2) / 6 + 1;
//#endif
//#ifndef OMEGA_FIELD
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


/* ------------------------------ PROBE DEFINES ------------------------------ */

constexpr int probe_x = (NX/2);
constexpr int probe_y = (NY/2);
constexpr int probe_z = (NZ_TOTAL/2);
constexpr int probe_index = probe_x + NX * (probe_y + NY*(probe_z));


/* --------------------------------- MACROS --------------------------------- */

#if defined(HO_RR) || defined(HOME_LBM)
    #define HIGH_ORDER_COLLISION
#endif


//#define COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
    #define HALO_SIZE 1
    const size_t VEL_GRAD_BLOCK_SIZE = (BLOCK_NX + 2 * HALO_SIZE) * (BLOCK_NY + 2 * HALO_SIZE) * (BLOCK_NZ + 2 * HALO_SIZE) * 3;
#else
    const size_t VEL_GRAD_BLOCK_SIZE = 0;
#endif

#ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
    #define HALO_SIZE 1
    const size_t CONFORMATION_GRAD_BLOCK_SIZE = (BLOCK_NX + 2 * HALO_SIZE) * (BLOCK_NY + 2 * HALO_SIZE) * (BLOCK_NZ + 2 * HALO_SIZE) * 6;
#else
    const size_t CONFORMATION_GRAD_BLOCK_SIZE = 0;
#endif

constexpr int MAX_SHARED_MEMORY_SIZE = myMax(BLOCK_LBM_SIZE_POP, myMax(VEL_GRAD_BLOCK_SIZE, CONFORMATION_GRAD_BLOCK_SIZE));

//FUNCTION DECLARATION MACROS
#ifdef DYNAMIC_SHARED_MEMORY
    #define DYNAMIC_SHARED_MEMORY_PARAMS ,SHARED_MEMORY_SIZE
#else
    #define DYNAMIC_SHARED_MEMORY_PARAMS
#endif


// Single-pointer macros
#ifdef DENSITY_CORRECTION
    #define DENSITY_CORRECTION_PARAMS_DECLARATION(PREFIX) dfloat *PREFIX##mean_rho,
    #define DENSITY_CORRECTION_PARAMS(PREFIX) PREFIX##mean_rho,
#else
    #define DENSITY_CORRECTION_PARAMS_DECLARATION(PREFIX)
    #define DENSITY_CORRECTION_PARAMS(PREFIX)
#endif



#ifdef BC_FORCES
    #define BC_FORCES_PARAMS_DECLARATION(PREFIX) dfloat *PREFIX##BC_Fx, dfloat *PREFIX##BC_Fy, dfloat *PREFIX##BC_Fz,
    #define BC_FORCES_PARAMS(PREFIX) PREFIX##BC_Fx, PREFIX##BC_Fy, PREFIX##BC_Fz,
#else
    #define BC_FORCES_PARAMS_DECLARATION(PREFIX)
    #define BC_FORCES_PARAMS(PREFIX)
#endif

#ifdef PARTICLE_TRACER
    #define PARTICLE_TRACER_PARAMS_DECLARATION(PREFIX) dfloat* PREFIX##particlePos
    #define PARTICLE_TRACER_PARAMS_PTR(PREFIX) PREFIX##particlePos
#else
    #define PARTICLE_TRACER_PARAMS_DECLARATION(PREFIX)
    #define PARTICLE_TRACER_PARAMS_PTR(PREFIX)
#endif




#if NODE_TYPE_SAVE
    #define NODE_TYPE_SAVE_PARAMS_DECLARATION dfloat *nodeTypeSave,
    #define NODE_TYPE_SAVE_PARAMS nodeTypeSave,
#else
    #define NODE_TYPE_SAVE_PARAMS_DECLARATION
    #define NODE_TYPE_SAVE_PARAMS
#endif



#ifdef OMEGA_FIELD
    #define OMEGA_FIELD_PARAMS_DECLARATION dfloat *omega,
    #define OMEGA_FIELD_PARAMS omega,
#else
    #define OMEGA_FIELD_PARAMS_DECLARATION
    #define OMEGA_FIELD_PARAMS
#endif



// Double-pointer macros 
#ifdef OMEGA_FIELD
    #define OMEGA_FIELD_PARAMS_DECLARATION_PTR ,dfloat** omega
    #define OMEGA_FIELD_PARAMS_PTR ,&omega
#else
    #define OMEGA_FIELD_PARAMS_DECLARATION_PTR
    #define OMEGA_FIELD_PARAMS_PTR
#endif



#ifdef SECOND_DIST
    #define SECOND_DIST_PARAMS_DECLARATION_PTR ,dfloat** C
    #define SECOND_DIST_PARAMS_PTR ,&C
#else
    #define SECOND_DIST_PARAMS_DECLARATION_PTR
    #define SECOND_DIST_PARAMS_PTR
#endif


#ifdef A_XX_DIST
    #define A_XX_DIST_PARAMS_DECLARATION_PTR ,dfloat** Axx
    #define A_XX_DIST_PARAMS_PTR ,&Axx
#else
    #define A_XX_DIST_PARAMS_DECLARATION_PTR
    #define A_XX_DIST_PARAMS_PTR
#endif

#ifdef A_XY_DIST
    #define A_XY_DIST_PARAMS_DECLARATION_PTR ,dfloat** Axy
    #define A_XY_DIST_PARAMS_PTR ,&Axy
#else
    #define A_XY_DIST_PARAMS_DECLARATION_PTR
    #define A_XY_DIST_PARAMS_PTR
#endif


#ifdef A_XZ_DIST
    #define A_XZ_DIST_PARAMS_DECLARATION_PTR ,dfloat** Axz
    #define A_XZ_DIST_PARAMS_PTR ,&Axz
#else
    #define A_XZ_DIST_PARAMS_DECLARATION_PTR
    #define A_XZ_DIST_PARAMS_PTR
#endif

#ifdef A_YY_DIST
    #define A_YY_DIST_PARAMS_DECLARATION_PTR ,dfloat** Ayy
    #define A_YY_DIST_PARAMS_PTR ,&Ayy
#else
    #define A_YY_DIST_PARAMS_DECLARATION_PTR
    #define A_YY_DIST_PARAMS_PTR
#endif

#ifdef A_YZ_DIST
    #define A_YZ_DIST_PARAMS_DECLARATION_PTR ,dfloat** Ayz
    #define A_YZ_DIST_PARAMS_PTR ,&Ayz
#else
    #define A_YZ_DIST_PARAMS_DECLARATION_PTR
    #define A_YZ_DIST_PARAMS_PTR
#endif

#ifdef A_ZZ_DIST
    #define A_ZZ_DIST_PARAMS_DECLARATION_PTR ,dfloat** Azz
    #define A_ZZ_DIST_PARAMS_PTR ,&Azz
#else
    #define A_ZZ_DIST_PARAMS_DECLARATION_PTR
    #define A_ZZ_DIST_PARAMS_PTR
#endif




#ifdef PARTICLE_TRACER
    #define PARTICLE_TRACER_PARAMS_DECLARATION_PTR(PREFIX) , dfloat** PREFIX##particlePos
    #define PARTICLE_TRACER_PARAMS_PTR(PREFIX) , &PREFIX##particlePos
#else
    #define PARTICLE_TRACER_PARAMS_DECLARATION_PTR(PREFIX)
    #define PARTICLE_TRACER_PARAMS_PTR(PREFIX)
#endif



#if MEAN_FLOW
    #define MEAN_FLOW_PARAMS_DECLARATION_PTR ,dfloat** m_fMom,dfloat** m_rho,dfloat** m_ux,dfloat** m_uy,dfloat** m_uz
    #define MEAN_FLOW_PARAMS_PTR , &m_fMom, &m_rho, &m_ux, &m_uy, &m_uz
    #ifdef SECOND_DIST
        #define MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR ,dfloat** m_c
        #define MEAN_FLOW_SECOND_DIST_PARAMS_PTR , &m_c
    #else
        #define MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR
        #define MEAN_FLOW_SECOND_DIST_PARAMS_PTR
    #endif
#else
    #define MEAN_FLOW_PARAMS_DECLARATION_PTR
    #define MEAN_FLOW_PARAMS_PTR
    #define MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR
    #define MEAN_FLOW_SECOND_DIST_PARAMS_PTR
#endif



#ifdef BC_FORCES
    #define BC_FORCES_PARAMS_DECLARATION_PTR(PREFIX) ,dfloat** PREFIX##BC_Fx ,dfloat** PREFIX##BC_Fy ,dfloat** PREFIX##BC_Fz
    #define BC_FORCES_PARAMS_PTR(PREFIX) ,&PREFIX##BC_Fx ,&PREFIX##BC_Fy ,&PREFIX##BC_Fz
#else
    #define BC_FORCES_PARAMS_DECLARATION_PTR(PREFIX)
    #define BC_FORCES_PARAMS_PTR(PREFIX)
#endif


#ifdef DENSITY_CORRECTION
    #define DENSITY_CORRECTION_PARAMS_DECLARATION_PTR(PREFIX) ,dfloat** PREFIX##mean_rho
    #define DENSITY_CORRECTION_PARAMS_PTR(PREFIX) , &PREFIX##mean_rho
#else
    #define DENSITY_CORRECTION_PARAMS_DECLARATION_PTR
    #define DENSITY_CORRECTION_PARAMS_PTR
#endif


#endif //!__DEFINITIONS_H