#define PARTICLE_MODEL

constexpr dfloat RE = 500;

constexpr int SCALE = 1;
constexpr int N_STEPS = 100;
constexpr int N = 256 * SCALE;
constexpr int NX = N;        // size x of the grid 
constexpr int NY = N;        // size y of the grid
constexpr int NZ = N;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat U_MAX = 0.0;  
constexpr dfloat L = N;

constexpr dfloat FX = 0.0;        // force in x
constexpr dfloat FY = 0.0;        // force in y
constexpr dfloat FZ = 1e-7;        // force in z (flow direction in most cases)

// value for the velocity initial condition in the domain
constexpr dfloat U_0_X = 0.0;
constexpr dfloat U_0_Y = 0.0;
constexpr dfloat U_0_Z = 0.0;
constexpr dfloat RHO_0 = 1.0;

// values options for boundary conditions //not used yet
__device__ const dfloat UX_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};


    #define BC_X_PERIODIC
    #define BC_Y_PERIODIC
    #define BC_Z_PERIODIC

constexpr dfloat VISC = U_MAX * NX / RE;
constexpr dfloat TAU = 0.8;     // relaxation time
constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;

#ifdef THERMAL_MODEL
    #define SECOND_DIST

    #define D3G19

    constexpr bool T_BOUYANCY = true;
    constexpr dfloat T_PR_NUMBER = 10; //Prandtl Number
    constexpr dfloat T_RA_NUMBER = 1e5; // Rayleigh Number
    constexpr dfloat BN_NUMBER = 0.2*8;
    constexpr dfloat T_GR_NUMBER = T_RA_NUMBER/T_PR_NUMBER; //Grashof number

    constexpr dfloat T_DELTA_T = 1; //temperature difference
    constexpr dfloat T_REFERENCE  = 1; //better closer to 1
    constexpr dfloat T_COLD = T_REFERENCE - T_DELTA_T/2.0;
    constexpr dfloat T_HOT = T_REFERENCE + T_DELTA_T/2.0;
    constexpr dfloat C_0 = T_REFERENCE; //initial temperature field 

    constexpr dfloat T_DIFFUSIVITY = VISC/T_PR_NUMBER; // alpha

    constexpr dfloat T_gravity_t_beta = 0;
    constexpr dfloat Ra_conf = 0;

    __device__ const dfloat gravity_vector[3] =  {1.0, 0.0, 0.0};

    constexpr dfloat T_DIFF_REF = (1.0/6);
    constexpr dfloat G_TAU = T_DIFF_REF*3+0.5;
    constexpr dfloat G_OMEGA  = 1.0/G_TAU;
    constexpr dfloat G_TT_OMEGA = 1.0-0.5*G_OMEGA;

    constexpr dfloat T_DIFF_FLUC = T_DIFFUSIVITY - T_DIFF_REF;
    constexpr dfloat G_AAA = (3*T_DIFF_FLUC*G_OMEGA);
    constexpr dfloat G_DIFF_FLUC_COEF = G_AAA/(1.0+G_AAA);
#endif //THERMAL_MODEL


/* --------------------- INITIALIZATION LOADING DEFINES -------------------- */
constexpr int INI_STEP = 0;
#define LOAD_CHECKPOINT false   // loads simulation checkpoint from folder 
                                // (folder name defined below)


// Folder with simulation to load data from last checkpoint. 
// WITHOUT ID_SIM (change it in ID_SIM) AND "/" AT THE END
#define SIMULATION_FOLDER_LOAD_CHECKPOINT "TEST"
// Interval to make checkpoint to save all simulation data and restart from it.
// It must not be very frequent (10000 or more), because it takes a long time
#define CHECKPOINT_SAVE false // the frequency on which the simulation checkpoint is saved
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0

#ifdef PARTICLE_MODEL
    #define IBM_METHOD
    #define DEM_METHOD
    #define LOCAL_FORCES
    //number of particles
    #define NUM_PARTICLES 2

    constexpr dfloat GX = 0.0;
    constexpr dfloat GY = 0.0;
    constexpr dfloat GZ = 0.0;

    constexpr dfloat FLUID_DENSITY = 1.0;


    #ifdef DEM_METHOD
        constexpr dfloat PP_FRICTION_COEF = 0.1; // friction coeficient particle particle -- case
        constexpr dfloat PW_FRICTION_COEF = 0.1; // friction coeficient particle wall -- case
        constexpr dfloat PP_REST_COEF = 0.9; // restitution coeficient particle particle -- case
        constexpr dfloat PW_REST_COEF = 0.9; // restitution coeficient particle wall-- case
        //#define REST_COEF_CORRECTION


        //material properties
        constexpr dfloat PARTICLE_YOUNG_MODULUS = 0.5; //-- case
        constexpr dfloat PARTICLE_POISSON_RATIO = 0.3; //-- case


        constexpr dfloat WALL_YOUNG_MODULUS = 100.0; //-- case
        constexpr dfloat WALL_POISSON_RATIO = 0.3; //-- case

    #endif //DEM_METHOD
#endif //PARTICLE_MODEL

/* -------------------------- BOUNDARY CONDITIONS -------------------------- */

// --- X direction ---
//#define IBM_BC_X_WALL

#ifdef BC_X_WALL
    // TODO: not implemented yet
    #define BC_X_WALL_UY 0.0
    #define BC_X_WALL_UZ 0.0
#endif //IBM_BC_X_WALL

#ifdef BC_X_PERIODIC
    #define BC_X_0 (0)
    #define BC_X_E (NX-0)
#endif //BC_X_PERIODIC

// --- Y direction ---
// #define BC_Y_WALL
//#define IBM_BC_Y_PERIODIC

#ifdef BC_Y_WALL
    // TODO: not implemented yet
    #define BC_Y_WALL_UX 0.0
    #define BC_Y_WALL_UZ 0.0
#endif //BC_Y_WALL

#ifdef BC_Y_PERIODIC
    #define BC_Y_0 0
    #define BC_Y_E (NY-0)
#endif //BC_Y_PERIODIC

// --- Z direction ---
//#define IBM_BC_Z_WALL
#define IBM_BC_Z_PERIODIC

#ifdef BC_Z_WALL
    // TODO: not implemented yet
    #define BC_Z_WALL_UX 0.0
    #define BC_Z_WALL_UY 0.0
#endif //BC_Z_WALL
#ifdef BC_Z_PERIODIC
    //TODO: FIX with multi-gpu, it currently does not work with values different than 0 and NZ_TOTAl
    #define BC_Z_0 0
    #define BC_Z_E (NZ_TOTAL-0)
#endif //BC_Z_PERIODIC
