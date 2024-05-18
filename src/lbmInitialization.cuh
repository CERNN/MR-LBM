#ifndef __LBM_INITIALIZATION_CUH
#define __LBM_INITIALIZATION_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "var.h"
#include "nodeTypeMap.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

/*
*   @brief Initializes random numbers (useful to initialize turbulence)
*   @param randomNumbers: vector of random numbers (size is NX*NY*NZ)
*   @param seed: seed to use for generation
*/
__host__
void initializationRandomNumbers(
    float* randomNumbers, 
    int seed
);


/*
*   @brief Initializes moments with equilibrium population, with density 
*          and velocity defined in the function itself
*   @param fMom: moments to be inialized to be initialized 
*/
__global__ void gpuInitialization_mom(
    dfloat *fMom, float* randomNumbers);

/*
*   @brief Initializes populations in the intefaces based on the moments 
*          defined in the gpuInitialization_mom
*   @param fMom: moments used to initialize the interface populations
*   @param fGhostX_0: populations for threadIdx.x == 0
*   @param fGhostX_1: populations for threadIdx.x == NX-1
*   @param fGhostY_0: populations for threadIdx.y == 0
*   @param fGhostY_1: populations for threadIdx.y == NY-1
*   @param fGhostZ_0: populations for threadIdx.z == 0
*   @param fGhostZ_1: populations for threadIdx.z == NZ-1
*/
__global__ void gpuInitialization_pop(
    dfloat *fMom,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1
    #ifdef SECOND_DIST 
    ,dfloat *g_fGhostX_0, dfloat *g_fGhostX_1,
    dfloat *g_fGhostY_0, dfloat *g_fGhostY_1,
    dfloat *g_fGhostZ_0, dfloat *g_fGhostZ_1
    #endif 
    );

/*
*   @brief Initialize the boundary condition node type
*   @param nodeType: node type ID
*/
__global__ void gpuInitialization_nodeType(
    unsigned int *dNodeType);

/*
*   @brief Initialize the boundary condition node type
*   @param nodeType: node type ID
*/
__host__ void hostInitialization_nodeType_bulk(
    unsigned int *hNodeType);


/*
*   @brief Initialize the boundary condition node type
*   @param nodeType: node type ID
*/
__host__ void hostInitialization_nodeType(
    unsigned int *hNodeType);


/*
*   @brief Initialize the local forces
*   @param d_L_Fx: local force in the x direction
*   @param d_L_Fy: local force in the y direction
*   @param d_L_Fz: local force in the z direction
*/
__global__ void gpuInitialization_force(
    dfloat *d_BC_Fx, dfloat* d_BC_Fy, dfloat* d_BC_Fz);   

/*
*   @brief Initializes boundary conditions based on csv with voxels coordinates
*   @param filename: csv filename
*   @param dNodeType: nodeType arrary
*/
__host__ void read_xyz_file(
    const std::string& filename, 
    unsigned int *dNodeType);


/*
*   @brief determines if the lattice is fluid or solid, calls bc_id to define the bc id;
*   @param dNodeType: nodeType arrary\
*/
__global__ void define_voxel_bc(unsigned int *dNodeType);


/*
 *   @brief defines the bc id based on the neighboring lattices;
 *   @param dNodeType: nodeType arrary
 *   @param x: x coordinate of the lattice
 *   @param y: y coordinate of the lattice
 *   @param y: z coordinate of the lattice
 */
__host__ __device__
unsigned int bc_id(unsigned int *dNodeType, int x,int y,int z);


#endif // !__LBM_INITIALIZATION_CUH