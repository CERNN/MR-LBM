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


#ifdef parallelPlatesBounceBack_
    #define BC_X_PERIODIC
    #define BC_Y_WALL
    #define BC_Z_PERIODIC
#endif

#ifdef squaredDuct_
    #define BC_X_WALL
    #define BC_Y_WALL
    #define BC_Z_PERIODIC
#endif

#ifdef closeCavity_
    #define BC_X_WALL
    #define BC_Y_WALL
    #define BC_Z_WALL
#endif

#ifdef taylorGreen_
    #define BC_X_PERIODIC
    #define BC_Y_PERIODIC
    #define BC_Z_PERIODIC
#endif

#ifdef lidDrivenCavity_
    #define BC_X_WALL
    #define BC_Y_WALL
    #define BC_Z_PERIODIC
#endif

#ifdef lidDrivenCavity_3D_
    #define BC_X_WALL
    #define BC_Y_WALL
    #define BC_Z_WALL
#endif

#ifdef benchmark_
    #define BC_X_PERIODIC
    #define BC_Y_PERIODIC
    #define BC_Z_PERIODIC
#endif

#ifdef lidDrivenCavity_3D_new_
    #define BC_X_WALL
    #define BC_Y_WALL
    #define BC_Z_WALL
#endif

#ifdef voxel_
    #define BC_X_PERIODIC
    #define BC_Y_PERIODIC
    #define BC_Z_PERIODIC
#endif

#ifdef expansion_
    #define BC_X_PERIODIC
    #define BC_Y_WALL
    #define BC_Z_PERIODIC
#endif

#ifdef externalFlow_
    #define BC_X_PERIODIC // free slip
    #define BC_Y_PERIODIC // free slep
    #define BC_Z_WALL// inlet outet
#endif

#ifdef jetFlow_
    #define BC_X_PERIODIC // free slip
    #define BC_Y_PERIODIC // free slep
    #define BC_Z_WALL// inlet outet
#endif




#ifdef BC_X_WALL
    #define INTERFACE_BC_WEST   INTERFACE_BC_WEST_BLOCK
    #define INTERFACE_BC_EAST   INTERFACE_BC_EAST_BLOCK
#endif
#ifdef BC_X_PERIODIC
    #define INTERFACE_BC_WEST   INTERFACE_BC_WEST_PERIO
    #define INTERFACE_BC_EAST   INTERFACE_BC_EAST_PERIO
#endif

#ifdef BC_Y_WALL
    #define INTERFACE_BC_SOUTH  INTERFACE_BC_SOUTH_BLOCK
    #define INTERFACE_BC_NORTH  INTERFACE_BC_NORTH_BLOCK
#endif
#ifdef BC_Y_PERIODIC
    #define INTERFACE_BC_SOUTH  INTERFACE_BC_SOUTH_PERIO
    #define INTERFACE_BC_NORTH  INTERFACE_BC_NORTH_PERIO
#endif

#ifdef BC_Z_WALL
    #define INTERFACE_BC_FRONT  INTERFACE_BC_FRONT_BLOCK
    #define INTERFACE_BC_BACK   INTERFACE_BC_BACK_BLOCK
#endif
#ifdef BC_Z_PERIODIC
    #define INTERFACE_BC_FRONT  INTERFACE_BC_FRONT_PERIO
    #define INTERFACE_BC_BACK   INTERFACE_BC_BACK_PERIO
#endif




#endif