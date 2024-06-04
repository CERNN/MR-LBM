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

/* ----------------------------- PROBLEM DEFINE ---------------------------- */

#define BC_PROBLEM rayleighBenard_2D

/* --------------------------- COLLISION METHOD  ------------------------- */
#define MR_LBM
//#define HO_RR //http://dx.doi.org/10.1063/1.4981227
//#define HOME_LBM //https://inria.hal.science/hal-04223237/


/* --------------------- NON-NEWTONIAN FLUID DEFINES ------------------- */
//#define BINGHAM

/* --------------------------- LES DEFINITIONS TYPE ------------------------- */
// Uncomment the one to use. Comment all to simulate newtonian fluid
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY //https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------------- OTHER DEFINITIONS ------------------------- */
//#define DENSITY_CORRECTION


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
#define ONLY_FINAL_MACRO (true) //save only the last time step macroscopic
#define MACR_SAVE (1000)
#define RANDOM_NUMBERS false    // to generate random numbers 
                                // (useful for turbulence)

//#define BC_FORCES //create scalar field to export the reaction forces from BC;
//#define SAVE_BC_FORCES // define if it will export BC force field to bin

constexpr bool console_flush = false;

//#define PARTICLE_TRACER  // define if will traces massless particles inside the flow
#define PARTICLE_TRACER_SAVE false

/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
constexpr int INI_STEP = 0; // initial simulation step (0 default)
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
                                // (folder name defined below)


// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
/* ------------------------------------------------------------------------- */



#define GPU_INDEX 0
/* --------------------------  SIMULATION DEFINES -------------------------- */

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)


#define CASE_DIRECTORY cases
#define CASE_CONSTANTS STR(CASE_DIRECTORY/BC_PROBLEM/constants)
#define CASE_BC_INIT STR(CASE_DIRECTORY/BC_PROBLEM/constants)
#define CASE_BC_DEF STR(CASE_DIRECTORY/BC_PROBLEM/bc_definition)
#define CASE_FLOW_INITIALIZATION STR(CASE_DIRECTORY/BC_PROBLEM/flow_initialization)


#include CASE_CONSTANTS


#include "definitions.h"
#endif //__VAR_H