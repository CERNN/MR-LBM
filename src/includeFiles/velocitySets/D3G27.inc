constexpr unsigned char GQ = 19;        // number of velocities
constexpr unsigned char GF = 5;         // number of velocities on each face
constexpr dfloat gW0 = 8.0 / 27;        // weight dist 0 population  (0, 0, 0)
constexpr dfloat gW1 = 2.0 / 27;        // weight dist 1 populations (1, 0, 0)
constexpr dfloat gW2 = 1.0 / 54;        // weight dist 2 populations (1, 1, 0)
constexpr dfloat gW3 = 1.0 / 216;       // weight dist 3 populations (1, 1, 1)
// velocities weight vector
__device__ const dfloat gw[GQ] = { 
    gW0,
    gW1, gW1, gW1, gW1, gW1, gW1, 
    gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2,
    gW3, gW3, gW3, gW3, gW3, gW3, gW3, gW3
};

constexpr dfloat g_as2 = 3.0;
constexpr dfloat g_cs2 = 1.0/g_as2;

// populations velocities vector 0 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
__device__ const char gcx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1};
__device__ const char gcy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1};
__device__ const char gcz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1};
