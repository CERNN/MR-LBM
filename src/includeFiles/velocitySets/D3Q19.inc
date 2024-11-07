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

constexpr dfloat F_M_0_SCALE = 1.0;
constexpr dfloat F_M_I_SCALE = as2;
constexpr dfloat F_M_II_SCALE = as2*as2/2;
constexpr dfloat F_M_IJ_SCALE = as2*as2;
