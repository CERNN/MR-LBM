constexpr unsigned char GQ = 7;        // number of velocities
constexpr unsigned char GF = 1;         // number of velocities on each face
constexpr dfloat gW0 = 1.0 / 4.0;         // population 0 weight (0, 0, 0)
constexpr dfloat gW1 = 1.0 / 8.0;        // adjacent populations (1, 0, 0)
// velocities weight vector
__device__ const dfloat gw[GQ] = { 
    gW0,
    gW1, gW1, gW1, gW1, gW1, gW1};

constexpr dfloat g_as2 = 4.0;
constexpr dfloat g_cs2 = 1.0/g_as2;

// populations velocities vector
__device__ const char gcx[GQ] = { 0, 1,-1, 0, 0, 0, 0};
__device__ const char gcy[GQ] = { 0, 0, 0, 1,-1, 0, 0};
__device__ const char gcz[GQ] = { 0, 0, 0, 0, 0, 1,-1};