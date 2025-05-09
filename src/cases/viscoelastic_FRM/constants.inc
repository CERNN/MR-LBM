#define FENE_P
#define TREAT_DATA_INCLUDE

constexpr dfloat RE = 1;

constexpr int SCALE = 1;
constexpr int N = 128 * SCALE;
constexpr int NX = N;        // size x of the grid 
constexpr int NY = N;        // size y of the grid
constexpr int NZ = 4;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat K_const = 2.0*M_PI/(dfloat)N;
constexpr dfloat U_MAX = 0.004908875*2;  
constexpr dfloat SUM_VISC = U_MAX/(RE*K_const);
constexpr dfloat BETA = 2.0/3.0;  //solution (sum) = 1,2, solver = 0.8  so ratio is 0.8/1.2 //Thomases2007
constexpr dfloat VISC = BETA * SUM_VISC;
constexpr dfloat TAU = 0.5 + 3.0*VISC;     // relaxation time

constexpr dfloat F_0 = 2*U_MAX*VISC*K_const*K_const;
constexpr dfloat t_c = 2*VISC*K_const/F_0;

constexpr int CONFORMATION_SWITCH = 10*(int)t_c;
constexpr int N_STEPS = CONFORMATION_SWITCH + 50*(int)t_c;

constexpr dfloat FX = 0.0;        // force in x
constexpr dfloat FY = 0.0;        // force in y
constexpr dfloat FZ = 0.0;        // force in z (flow direction in most cases)

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

constexpr dfloat L = N;

constexpr dfloat MACH_NUMBER = U_MAX/0.57735026918962;


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
#define LOAD_FIELD false
#define LOAD_FIELD_STEP 0

#ifdef FENE_P
    #define OLDROYD_B
    #define CONVECTION_DIFFUSION_TRANSPORT
    #define D3G19

    #define CONFORMATION_TENSOR    
    #define A_XX_DIST
    #define A_XY_DIST
    #define A_XZ_DIST
    #define A_YY_DIST
    #define A_YZ_DIST
    #define A_ZZ_DIST

    #define COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
    #define COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE


    constexpr dfloat CONF_ZERO = (dfloat)(10.0);

    constexpr dfloat nu_p = (VISC/BETA) - VISC;


    constexpr dfloat Weissenberg_number = 0.6;



    constexpr dfloat LAMBDA = Weissenberg_number*t_c;
    constexpr dfloat inv_lambda = 1.0/LAMBDA;
    constexpr dfloat fenep_re = 10.0;

    //ADVECTION-DIFFUSIONS CONSTANTS
    constexpr dfloat CONF_DIFFUSIVITY_RATIO = 1000; //OR Pr in malaspinas
    constexpr dfloat CONF_DIFFUSIVITY = VISC/CONF_DIFFUSIVITY_RATIO; // a0lpha

    constexpr dfloat CONF_DIFF_REF = (1.0/6);
    constexpr dfloat CONF_TAU = CONF_DIFF_REF*3+0.5;
    constexpr dfloat CONF_OMEGA  = 1.0/CONF_TAU;
    constexpr dfloat CONF_TT_OMEGA = 1.0-0.5*CONF_OMEGA;

    constexpr dfloat CONF_DIFF_FLUC = CONF_DIFFUSIVITY - CONF_DIFF_REF;
    constexpr dfloat CONF_AAA = (3*CONF_DIFF_FLUC*CONF_OMEGA);
    constexpr dfloat CONF_DIFF_FLUC_COEF = CONF_AAA/(1.0+CONF_AAA);
#endif
