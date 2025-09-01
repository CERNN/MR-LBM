/**
*   @file lbmInitialization.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @brief Variable Initialization
*   @version 0.4.0
*   @date 01/09/2025
*/


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
    dfloat *fMom, dfloat* randomNumbers);

/*
*   @brief Initializes populations in the intefaces based on the moments 
*          defined in the gpuInitialization_mom
*   @param fMom: moments used to initialize the interface populations
*   @param ghostInterface interface block transfer information
*/
__global__ void gpuInitialization_pop(
    dfloat *fMom,  ghostInterfaceData ghostInterface);

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