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


constexpr dfloat F_M_0_SCALE = 1.0;
constexpr dfloat F_M_I_SCALE = as2;
constexpr dfloat F_M_II_SCALE = as2*as2/2;
constexpr dfloat F_M_IJ_SCALE = as2*as2;