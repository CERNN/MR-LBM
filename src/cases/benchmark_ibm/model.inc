//velocity set
#define D3Q19

/* --------------------------- COLLISION METHOD  ------------------------- */
#define COLLISION_TYPE MR_LBM
//#define COLLISION_TYPE HO_RR //http://dx.doi.org/10.1063/1.4981227
//#define COLLISION_TYPE HOME_LBM //https://inria.hal.science/hal-04223237/

/* --------------------------- LES DEFINITIONS TYPE ------------------------- */
// Uncomment the one to use. Comment all to simulate newtonian fluid
//#define LES_MODEL
//#define MODEL_CONST_SMAGORINSKY //https://doi.org/10.1016/j.jcp.2005.03.022

/* --------------------------- OTHER DEFINITIONS ------------------------- */
//#define DENSITY_CORRECTION

#define RANDOM_NUMBERS true    // to generate random numbers 
                               // (useful for turbulence)



// Change to location of nodes http://dx.doi.org/10.1016/j.jcp.2012.02.026
#define BREUGEM_PARAMETER (0.0)
// Mesh scale for IBM, minimum distance between nodes (lower, more nodes in particle)
#define MESH_SCALE 1.0
// Number of iterations of Coulomb algorithm to optimize the nodes positions
#define MESH_COULOMB 0
// #define ROTATION_LOCK true
// Assumed boundary thickness for IBM
#define IBM_THICKNESS 1.0
#define STENCIL_4 // Peskin Stencil
