#define BINGHAM
#define TREAT_DATA_INCLUDE

constexpr int N_STEPS = 10000;

constexpr dfloat U_TAU = 0.00579;     // friction velocity
constexpr int del = 96;               // integral lenght scale

constexpr unsigned int N = del;       // (32 multiple for better performance)
constexpr unsigned int NX = 2*del;    // size x of the grid 
constexpr unsigned int NY = 2*del;    // size y of the grid
constexpr unsigned int NZ = 16*del; //4*del/dZ;    // size z of the grid
constexpr int NZ_TOTAL = NZ;       // size z of the grid

constexpr dfloat U_MAX = U_TAU;       // max velocity
constexpr dfloat RHO_0 = 1.0;         // initial rho

constexpr dfloat RE = 180.0;	  // Reynolds number
constexpr dfloat Bn = 0.1;	      // Bingham number (0 < Bi < 0.5)

constexpr dfloat VISC = del*U_TAU/RE;// kinematic viscosity for Re_tau = 180
constexpr dfloat TAU = 3.0*VISC + 0.5;// relaxation time
constexpr dfloat S_Y = Bn*U_MAX*U_MAX; //yield-stress

constexpr dfloat L = N;

constexpr dfloat FX = 0.0;        // force in x
constexpr dfloat FY = 0.0;        // force in y
constexpr dfloat FZ = U_TAU*U_TAU/del;;        // force in z (flow direction in most cases)

// value for the velocity initial condition in the domain
constexpr dfloat U_0_X = 0.0;
constexpr dfloat U_0_Y = 0.0;
constexpr dfloat U_0_Z = 0.0;

// values options for boundary conditions //not used yet
__device__ const dfloat UX_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UY_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat UZ_BC[4] =  {0, 0, 0, 0};
__device__ const dfloat RHO_BC[4] = {RHO_0, RHO_0, RHO_0, RHO_0};


#define BC_X_PERIODIC
#define BC_Y_WALL
#define BC_Z_PERIODIC

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