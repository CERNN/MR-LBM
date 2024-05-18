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

//velocity set
#define D3Q19

/* ----------------------------- BC DEFINES ---------------------------- */

//#define parallelPlates_
//#define BC_PROBLEM parallelPlates
//#define lidDrivenCavity_3D_
//#define BC_PROBLEM lidDrivenCavity_3D

#define externalFlow_
#define BC_PROBLEM externalFlow
//#define voxel_
//#define BC_PROBLEM voxel
#define VOXEL_FILENAME "sphere_D32N128.csv"

//#define benchmark_
//#define BC_PROBLEM benchmark

#define BC_MOMENT_BASED

//#define DENSITY_CORRECTION

//#define HO_RR //http://dx.doi.org/10.1063/1.4981227
//#define HOME_LBM //https://inria.hal.science/hal-04223237/


/* --------------------- NON-NEWTONIAN FLUID DEFINES ------------------- */
//#define BINGHAM

/* --------------------------- LES DEFINITIONS TYPE ------------------------- */
// Uncomment the one to use. Comment all to simulate newtonian fluid
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY //https://doi.org/10.1016/j.jcp.2005.03.022

/* ----------------------------- OUTPUT DEFINES ---------------------------- */
    #define ID_SIM "000"            // prefix for simulation's files
#define PATH_FILES "TEST"  // path to save simulation's files

#define TREATFIELD (false) //treat data over the entire field
#define TREATPOINT (false) //treat data in a single or several points
#define TREATLINE (false) //save the macro in a line
#define SAVEDATA (true) //save treat data
#define CONSOLEPRINT (false) // print the console the data is being saved
#define MEAN_FLOW (false) // store the mean flow of the domain (used to calculate turbulent statistics)
#define SAVE_BC (false) //save the bc conditions, usefull for drawing the surface
#define ONLY_FINAL_MACRO (false) //save only the last time step macroscopic

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

constexpr bool console_flush = false;

//#define PARTICLE_TRACER  // define if will traces massless particles inside the flow
#define PARTICLE_TRACER_SAVE false

//#define THERMAL_MODEL 


/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
constexpr int INI_STEP = 0; // initial simulation step (0 default)
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
                                // (folder name defined below)
#define RANDOM_NUMBERS false    // to generate random numbers 
                                // (useful for turbulence)

// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
/* ------------------------------------------------------------------------- */



#define GPU_INDEX 0
/* --------------------------  SIMULATION DEFINES -------------------------- */

constexpr int SCALE = 1;
constexpr dfloat RE = 1000;
#define MACR_SAVE (1024)


constexpr int N = 128 * SCALE;
constexpr int NX = 2*N;        // size x of the grid 
                                    // (32 multiple for better performance)
constexpr int NY = 2*N;        // size y of the grid
constexpr int NZ = 4*N;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat U_MAX = 0.1;  

constexpr dfloat L = NZ;
constexpr dfloat VISC = 32*U_MAX / RE;
constexpr dfloat Ct = (1.0/L)/(1.0/U_MAX);
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;

constexpr dfloat turn_over_time = L / U_MAX;
constexpr int N_STEPS = 5*((int)turn_over_time);
constexpr dfloat total_time = N_STEPS *Ct;


constexpr dfloat TAU = 0.5 + 3.0*VISC;     // relaxation time

constexpr dfloat RHO_0 = 1.0;         // initial rho

constexpr dfloat FX = 0.0;        // force in x
constexpr dfloat FY = 0.0;        // force in y
constexpr dfloat FZ = 0.0;        // force in z (flow direction in most cases)

#ifdef THERMAL_MODEL
#define SECOND_DIST

#define D3G7


constexpr dfloat T_PR_NUMBER = 1.0; //Prandtl Number
constexpr dfloat T_RA_NUMBER = 1000000.0; // Rayleigh Number
constexpr dfloat T_GR_NUMBER = T_RA_NUMBER/T_PR_NUMBER; //Grashof number

constexpr dfloat T_DELTA_T = 0.5; //temperature difference
constexpr dfloat T_COLD = 1.0 - T_DELTA_T/2.0;
constexpr dfloat T_HOT = 1.0 + T_DELTA_T/2.0;
constexpr dfloat T_REFERENCE  = (T_HOT+T_COLD)/2.0; //better closer to 1
constexpr dfloat C_0 = T_REFERENCE; //initial temperature field 


constexpr dfloat T_DIFFUSIVITY = VISC/T_PR_NUMBER; // alpha

constexpr dfloat T_gravity_t_beta = T_RA_NUMBER * T_DIFFUSIVITY /(T_DELTA_T*L*L*L);

constexpr dfloat G_TAU = T_DIFFUSIVITY*3+0.5;
constexpr dfloat G_OMEGA  = 1.0/G_TAU;
constexpr dfloat G_T_OMEGA  = 1.0-G_OMEGA;
constexpr dfloat G_TT_OMEGA = 1.0-0.5*G_OMEGA;

#endif //THERMAL_MODEL

// value for the velocity initial condition in the domain
constexpr dfloat U_0_X = 0.0;
constexpr dfloat U_0_Y = 0.0;
constexpr dfloat U_0_Z = U_MAX;

// values options for boundary conditions //not used yet
__device__ const dfloat UX_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};



#include "definitions.h"
#endif //__VAR_H