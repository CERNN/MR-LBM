#include "./../var.h"

#ifndef __INTERFACE_BC_H
#define __INTERFACE_BC_H

#define INTERFACE_BC_WEST_PERIO (threadIdx.x == 0) 
#define INTERFACE_BC_WEST_BLOCK (threadIdx.x == 0 && x!=0) 
#define INTERFACE_BC_EAST_PERIO (threadIdx.x == (BLOCK_NX - 1))
#define INTERFACE_BC_EAST_BLOCK (threadIdx.x == (BLOCK_NX - 1) && x!=(NX-1))

#define INTERFACE_BC_SOUTH_PERIO (threadIdx.y == 0)
#define INTERFACE_BC_SOUTH_BLOCK (threadIdx.y == 0 && y !=0)
#define INTERFACE_BC_NORTH_PERIO (threadIdx.y == (BLOCK_NY - 1))
#define INTERFACE_BC_NORTH_BLOCK (threadIdx.y == (BLOCK_NY - 1) && y!=(NY-1))

#define INTERFACE_BC_BACK_PERIO (threadIdx.z == 0)
#define INTERFACE_BC_BACK_BLOCK (threadIdx.z == 0 && z !=0)
#define INTERFACE_BC_FRONT_PERIO (threadIdx.z == (BLOCK_NZ - 1))
#define INTERFACE_BC_FRONT_BLOCK (threadIdx.z == (BLOCK_NZ - 1) && z!=(NZ-1))


#ifdef BC_X_WALL
    #define INTERFACE_BC_WEST   INTERFACE_BC_WEST_BLOCK
    #define INTERFACE_BC_EAST   INTERFACE_BC_EAST_BLOCK
#endif //BC_X_WALL
#ifdef BC_X_PERIODIC
    #define INTERFACE_BC_WEST   INTERFACE_BC_WEST_PERIO
    #define INTERFACE_BC_EAST   INTERFACE_BC_EAST_PERIO
#endif //BC_X_PERIODIC

#ifdef BC_Y_WALL
    #define INTERFACE_BC_SOUTH  INTERFACE_BC_SOUTH_BLOCK
    #define INTERFACE_BC_NORTH  INTERFACE_BC_NORTH_BLOCK
#endif //BC_Y_WALL
#ifdef BC_Y_PERIODIC
    #define INTERFACE_BC_SOUTH  INTERFACE_BC_SOUTH_PERIO
    #define INTERFACE_BC_NORTH  INTERFACE_BC_NORTH_PERIO
#endif //BC_Y_PERIODIC

#ifdef BC_Z_WALL
    #define INTERFACE_BC_FRONT  INTERFACE_BC_FRONT_BLOCK
    #define INTERFACE_BC_BACK   INTERFACE_BC_BACK_BLOCK
#endif //BC_Z_WALL
#ifdef BC_Z_PERIODIC
    #define INTERFACE_BC_FRONT  INTERFACE_BC_FRONT_PERIO
    #define INTERFACE_BC_BACK   INTERFACE_BC_BACK_PERIO
#endif //BC_Z_PERIODIC



// Define the periodic offsets based on the compilation flags
#if defined(BC_X_PERIODIC) && defined(BC_Y_PERIODIC) && defined(BC_Z_PERIODIC)
    // Case: All 3 periodic boundaries
    #define NUM_PERIODIC_DOMAIN_OFFSET 27
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {
        {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1},
        {-1, 0, -1},  {-1, 0, 0},  {-1, 0, 1},
        {-1, 1, -1},  {-1, 1, 0},  {-1, 1, 1},

        {0, -1, -1},  {0, -1, 0},  {0, -1, 1},
        {0, 0, -1},   {0, 0, 0},   {0, 0, 1},
        {0, 1, -1},   {0, 1, 0},   {0, 1, 1},

        {1, -1, -1},  {1, -1, 0},  {1, -1, 1},
        {1, 0, -1},   {1, 0, 0},   {1, 0, 1},
        {1, 1, -1},   {1, 1, 0},   {1, 1, 1}
    };

#elif defined(BC_X_PERIODIC) && defined(BC_Y_PERIODIC)
    // Case: X and Y periodic
    #define NUM_PERIODIC_DOMAIN_OFFSET 9
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {
        {-1, -1, 0}, {-1, 0, 0}, {-1, 1, 0},
        {0, -1, 0},  {0, 0, 0},  {0, 1, 0},
        {1, -1, 0},  {1, 0, 0},  {1, 1, 0}
    };

#elif defined(BC_X_PERIODIC) && defined(BC_Z_PERIODIC)
    // Case: X and Z periodic
    #define NUM_PERIODIC_DOMAIN_OFFSET 9
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {
        {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1},
        {0, 0, -1},  {0, 0, 0},  {0, 0, 1},
        {1, 0, -1},  {1, 0, 0},  {1, 0, 1}
    };

#elif defined(BC_Y_PERIODIC) && defined(BC_Z_PERIODIC)
    // Case: Y and Z periodic
    #define NUM_PERIODIC_DOMAIN_OFFSET 9
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {
        {0, -1, -1}, {0, -1, 0}, {0, -1, 1},
        {0, 0, -1},  {0, 0, 0},  {0, 0, 1},
        {0, 1, -1},  {0, 1, 0},  {0, 1, 1}
    };

#elif defined(BC_X_PERIODIC)
    // Case: X periodic only
    #define NUM_PERIODIC_DOMAIN_OFFSET 3
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {{-1, 0, 0}, {0, 0, 0}, {1, 0, 0}};

#elif defined(BC_Y_PERIODIC)
    // Case: Y periodic only
    #define NUM_PERIODIC_DOMAIN_OFFSET 3
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3]= {{0, -1, 0}, {0, 0, 0}, {0, 1, 0}};

#elif defined(BC_Z_PERIODIC)
    // Case: Z periodic only
    #define NUM_PERIODIC_DOMAIN_OFFSET 3
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {{0, 0, -1}, {0, 0, 0}, {0, 0, 1}};

#else
    // Case: No periodic boundaries
    #define NUM_PERIODIC_DOMAIN_OFFSET 1
    __device__ const int PERIODIC_DOMAIN_OFFSET[NUM_PERIODIC_DOMAIN_OFFSET][3] = {{0, 0, 0}};

#endif




#endif //!__INTERFACE_BC_H