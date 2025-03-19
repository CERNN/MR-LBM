#define TREAT_DATA_INCLUDE

constexpr dfloat RE = 1;

constexpr int SCALE = 1;
constexpr int N = 128 * SCALE;
constexpr int NX = N;        // size x of the grid 
constexpr int NY = N;        // size y of the grid
constexpr int NZ = N;        // size z of the grid in one GPU
constexpr int NZ_TOTAL = NZ;       // size z of the grid


constexpr dfloat U_MAX = 1.0/(125.0*M_PI); 
constexpr dfloat L = (dfloat)N / (2.0 * M_PI);

constexpr dfloat VISC = L*U_MAX / RE;
 
constexpr dfloat Cv = 1.0/U_MAX;
constexpr dfloat Ct = (1.0/L)/(1.0/U_MAX);
constexpr dfloat Cx = 1.0/L;
constexpr dfloat Cp = 1.0/(Cx * Ct * Ct);

constexpr int N_STEPS = (int)(20.0/Ct)+1;

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

constexpr dfloat TAU = 0.5 + 3.0*VISC;     // relaxation time
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