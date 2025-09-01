/**
*   @file main.cuh
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @author Ricardo de Souza
*   @brief Main routine
*   @version 0.4.0
*   @date 01/09/2025
*/


// main.cuh
#ifndef MAIN_CUH
#define MAIN_CUH

#include <stdio.h>
#include <stdlib.h>

// CUDA INCLUDE
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// FILE INCLUDES
#include "var.h"
#include "globalStructs.h"
#include "auxFunctions.cuh"
#include "treatData.cuh"


#ifdef PARTICLE_MODEL
    #include "./particles/class/Particle.cuh"
    #include "./particles/utils/particlesReport.cuh"
    #include "./particles/models/particleSim.cuh"
    #include "./particles/models/dem/collision/collisionDetection.cuh"
    #include "./particles/models/dem/particleMovement.cuh"
#endif

#ifdef OMEGA_FIELD
    #include "nnf.h"
#endif

#include "errorDef.h"
//#include "structs.h"
//#include "globalFunctions.h"
#include "lbmInitialization.cuh"
#include "mlbm.cuh"
#include "saveData.cuh"
#include "checkpoint.cuh"

/*
*   @brief Swaps the pointers of two dfloat variables.
*   @param pt1: reference to the first dfloat pointer to be swapped
*   @param pt2: reference to the second dfloat pointer to be swapped
*/
__host__ __device__
void interfaceSwap(dfloat* &pt1, dfloat* &pt2) {
    dfloat *temp = pt1;
    pt1 = pt2;
    pt2 = temp;
}

void initializeCudaEvents(cudaEvent_t &start, cudaEvent_t &stop, cudaEvent_t &start_step, cudaEvent_t &stop_step) {
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));
}


dfloat recordElapsedTime(cudaEvent_t &start_step, cudaEvent_t &stop_step, int step, int ini_step) {
    checkCudaErrors(cudaEventRecord(stop_step, 0));
    checkCudaErrors(cudaEventSynchronize(stop_step));
    
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_step, stop_step));
    elapsedTime *= 0.001;

    size_t nodesUpdatedSync = (step - ini_step) * NUMBER_LBM_NODES;
    dfloat MLUPS = (nodesUpdatedSync / 1e6) / elapsedTime;
    return MLUPS;
}

/*
*   @brief Frees the memory allocated for the ghost interface data.
*   @param ghostInterface: reference to the ghost interface data structure
*/
__host__
void interfaceFree(ghostInterfaceData &ghostInterface)
{
    cudaFree(ghostInterface.fGhost.X_0);
    cudaFree(ghostInterface.fGhost.X_1);
    cudaFree(ghostInterface.fGhost.Y_0);
    cudaFree(ghostInterface.fGhost.Y_1);
    cudaFree(ghostInterface.fGhost.Z_0);
    cudaFree(ghostInterface.fGhost.Z_1);

    cudaFree(ghostInterface.gGhost.X_0);
    cudaFree(ghostInterface.gGhost.X_1);
    cudaFree(ghostInterface.gGhost.Y_0);
    cudaFree(ghostInterface.gGhost.Y_1);
    cudaFree(ghostInterface.gGhost.Z_0);
    cudaFree(ghostInterface.gGhost.Z_1);

    #ifdef SECOND_DIST
        cudaFree(ghostInterface.g_fGhost.X_0);
        cudaFree(ghostInterface.g_fGhost.X_1);
        cudaFree(ghostInterface.g_fGhost.Y_0);
        cudaFree(ghostInterface.g_fGhost.Y_1);
        cudaFree(ghostInterface.g_fGhost.Z_0);
        cudaFree(ghostInterface.g_fGhost.Z_1);

        cudaFree(ghostInterface.g_gGhost.X_0);
        cudaFree(ghostInterface.g_gGhost.X_1);
        cudaFree(ghostInterface.g_gGhost.Y_0);
        cudaFree(ghostInterface.g_gGhost.Y_1);
        cudaFree(ghostInterface.g_gGhost.Z_0);
        cudaFree(ghostInterface.g_gGhost.Z_1);
    #endif
    #ifdef A_XX_DIST
        cudaFree(ghostInterface.Axx_fGhost.X_0);
        cudaFree(ghostInterface.Axx_fGhost.X_1);
        cudaFree(ghostInterface.Axx_fGhost.Y_0);
        cudaFree(ghostInterface.Axx_fGhost.Y_1);
        cudaFree(ghostInterface.Axx_fGhost.Z_0);
        cudaFree(ghostInterface.Axx_fGhost.Z_1);

        cudaFree(ghostInterface.Axx_gGhost.X_0);
        cudaFree(ghostInterface.Axx_gGhost.X_1);
        cudaFree(ghostInterface.Axx_gGhost.Y_0);
        cudaFree(ghostInterface.Axx_gGhost.Y_1);
        cudaFree(ghostInterface.Axx_gGhost.Z_0);
        cudaFree(ghostInterface.Axx_gGhost.Z_1);
    #endif
    #ifdef A_XY_DIST
        cudaFree(ghostInterface.Axy_fGhost.X_0);
        cudaFree(ghostInterface.Axy_fGhost.X_1);
        cudaFree(ghostInterface.Axy_fGhost.Y_0);
        cudaFree(ghostInterface.Axy_fGhost.Y_1);
        cudaFree(ghostInterface.Axy_fGhost.Z_0);
        cudaFree(ghostInterface.Axy_fGhost.Z_1);

        cudaFree(ghostInterface.Axy_gGhost.X_0);
        cudaFree(ghostInterface.Axy_gGhost.X_1);
        cudaFree(ghostInterface.Axy_gGhost.Y_0);
        cudaFree(ghostInterface.Axy_gGhost.Y_1);
        cudaFree(ghostInterface.Axy_gGhost.Z_0);
        cudaFree(ghostInterface.Axy_gGhost.Z_1);
    #endif
    #ifdef A_XZ_DIST
        cudaFree(ghostInterface.Axz_fGhost.X_0);
        cudaFree(ghostInterface.Axz_fGhost.X_1);
        cudaFree(ghostInterface.Axz_fGhost.Y_0);
        cudaFree(ghostInterface.Axz_fGhost.Y_1);
        cudaFree(ghostInterface.Axz_fGhost.Z_0);
        cudaFree(ghostInterface.Axz_fGhost.Z_1);

        cudaFree(ghostInterface.Axz_gGhost.X_0);
        cudaFree(ghostInterface.Axz_gGhost.X_1);
        cudaFree(ghostInterface.Axz_gGhost.Y_0);
        cudaFree(ghostInterface.Axz_gGhost.Y_1);
        cudaFree(ghostInterface.Axz_gGhost.Z_0);
        cudaFree(ghostInterface.Axz_gGhost.Z_1);
    #endif
    #ifdef A_YY_DIST
        cudaFree(ghostInterface.Ayy_fGhost.X_0);
        cudaFree(ghostInterface.Ayy_fGhost.X_1);
        cudaFree(ghostInterface.Ayy_fGhost.Y_0);
        cudaFree(ghostInterface.Ayy_fGhost.Y_1);
        cudaFree(ghostInterface.Ayy_fGhost.Z_0);
        cudaFree(ghostInterface.Ayy_fGhost.Z_1);

        cudaFree(ghostInterface.Ayy_gGhost.X_0);
        cudaFree(ghostInterface.Ayy_gGhost.X_1);
        cudaFree(ghostInterface.Ayy_gGhost.Y_0);
        cudaFree(ghostInterface.Ayy_gGhost.Y_1);
        cudaFree(ghostInterface.Ayy_gGhost.Z_0);
        cudaFree(ghostInterface.Ayy_gGhost.Z_1);
    #endif
    #ifdef A_YZ_DIST
        cudaFree(ghostInterface.Ayz_fGhost.X_0);
        cudaFree(ghostInterface.Ayz_fGhost.X_1);
        cudaFree(ghostInterface.Ayz_fGhost.Y_0);
        cudaFree(ghostInterface.Ayz_fGhost.Y_1);
        cudaFree(ghostInterface.Ayz_fGhost.Z_0);
        cudaFree(ghostInterface.Ayz_fGhost.Z_1);

        cudaFree(ghostInterface.Ayz_gGhost.X_0);
        cudaFree(ghostInterface.Ayz_gGhost.X_1);
        cudaFree(ghostInterface.Ayz_gGhost.Y_0);
        cudaFree(ghostInterface.Ayz_gGhost.Y_1);
        cudaFree(ghostInterface.Ayz_gGhost.Z_0);
        cudaFree(ghostInterface.Ayz_gGhost.Z_1);
    #endif
    #ifdef A_ZZ_DIST
        cudaFree(ghostInterface.Azz_fGhost.X_0);
        cudaFree(ghostInterface.Azz_fGhost.X_1);
        cudaFree(ghostInterface.Azz_fGhost.Y_0);
        cudaFree(ghostInterface.Azz_fGhost.Y_1);
        cudaFree(ghostInterface.Azz_fGhost.Z_0);
        cudaFree(ghostInterface.Azz_fGhost.Z_1);

        cudaFree(ghostInterface.Azz_gGhost.X_0);
        cudaFree(ghostInterface.Azz_gGhost.X_1);
        cudaFree(ghostInterface.Azz_gGhost.Y_0);
        cudaFree(ghostInterface.Azz_gGhost.Y_1);
        cudaFree(ghostInterface.Azz_gGhost.Z_0);
        cudaFree(ghostInterface.Azz_gGhost.Z_1);
    #endif
    /*
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
        cudaFree(ghostInterface.f_uGhost.X_0);
        cudaFree(ghostInterface.f_uGhost.X_1);
        cudaFree(ghostInterface.f_uGhost.Y_0);
        cudaFree(ghostInterface.f_uGhost.Y_1);
        cudaFree(ghostInterface.f_uGhost.Z_0);
        cudaFree(ghostInterface.f_uGhost.Z_1);

        cudaFree(ghostInterface.g_uGhost.X_0);
        cudaFree(ghostInterface.g_uGhost.X_1);
        cudaFree(ghostInterface.g_uGhost.Y_0);
        cudaFree(ghostInterface.g_uGhost.Y_1);
        cudaFree(ghostInterface.g_uGhost.Z_0);
        cudaFree(ghostInterface.g_uGhost.Z_1);
    #endif

    
    #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
        cudaFree(ghostInterface.conf_fGhost.X_0);
        cudaFree(ghostInterface.conf_fGhost.X_1);
        cudaFree(ghostInterface.conf_fGhost.Y_0);
        cudaFree(ghostInterface.conf_fGhost.Y_1);
        cudaFree(ghostInterface.conf_fGhost.Z_0);
        cudaFree(ghostInterface.conf_fGhost.Z_1);

        cudaFree(ghostInterface.conf_gGhost.X_0);
        cudaFree(ghostInterface.conf_gGhost.X_1);
        cudaFree(ghostInterface.conf_gGhost.Y_0);
        cudaFree(ghostInterface.conf_gGhost.Y_1);
        cudaFree(ghostInterface.conf_gGhost.Z_0);
        cudaFree(ghostInterface.conf_gGhost.Z_1);
    #endif
    */

    if (LOAD_CHECKPOINT){
        cudaFree(ghostInterface.h_fGhost.X_0);
        cudaFree(ghostInterface.h_fGhost.X_1);
        cudaFree(ghostInterface.h_fGhost.Y_0);
        cudaFree(ghostInterface.h_fGhost.Y_1);
        cudaFree(ghostInterface.h_fGhost.Z_0);
        cudaFree(ghostInterface.h_fGhost.Z_1);
        #ifdef SECOND_DIST
            cudaFree(ghostInterface.g_h_fGhost.X_0);
            cudaFree(ghostInterface.g_h_fGhost.X_1);
            cudaFree(ghostInterface.g_h_fGhost.Y_0);
            cudaFree(ghostInterface.g_h_fGhost.Y_1);
            cudaFree(ghostInterface.g_h_fGhost.Z_0);
            cudaFree(ghostInterface.g_h_fGhost.Z_1);
        #endif
        #ifdef A_XX_DIST
            cudaFree(ghostInterface.Axx_h_fGhost.X_0);
            cudaFree(ghostInterface.Axx_h_fGhost.X_1);
            cudaFree(ghostInterface.Axx_h_fGhost.Y_0);
            cudaFree(ghostInterface.Axx_h_fGhost.Y_1);
            cudaFree(ghostInterface.Axx_h_fGhost.Z_0);
            cudaFree(ghostInterface.Axx_h_fGhost.Z_1);
        #endif
        #ifdef A_XY_DIST
            cudaFree(ghostInterface.Axy_h_fGhost.X_0);
            cudaFree(ghostInterface.Axy_h_fGhost.X_1);
            cudaFree(ghostInterface.Axy_h_fGhost.Y_0);
            cudaFree(ghostInterface.Axy_h_fGhost.Y_1);
            cudaFree(ghostInterface.Axy_h_fGhost.Z_0);
            cudaFree(ghostInterface.Axy_h_fGhost.Z_1);
        #endif
        #ifdef A_XZ_DIST
            cudaFree(ghostInterface.Axz_h_fGhost.X_0);
            cudaFree(ghostInterface.Axz_h_fGhost.X_1);
            cudaFree(ghostInterface.Axz_h_fGhost.Y_0);
            cudaFree(ghostInterface.Axz_h_fGhost.Y_1);
            cudaFree(ghostInterface.Axz_h_fGhost.Z_0);
            cudaFree(ghostInterface.Axz_h_fGhost.Z_1);
        #endif
        #ifdef A_YY_DIST
            cudaFree(ghostInterface.Ayy_h_fGhost.X_0);
            cudaFree(ghostInterface.Ayy_h_fGhost.X_1);
            cudaFree(ghostInterface.Ayy_h_fGhost.Y_0);
            cudaFree(ghostInterface.Ayy_h_fGhost.Y_1);
            cudaFree(ghostInterface.Ayy_h_fGhost.Z_0);
            cudaFree(ghostInterface.Ayy_h_fGhost.Z_1);
        #endif
        #ifdef A_YZ_DIST
            cudaFree(ghostInterface.Ayz_h_fGhost.X_0);
            cudaFree(ghostInterface.Ayz_h_fGhost.X_1);
            cudaFree(ghostInterface.Ayz_h_fGhost.Y_0);
            cudaFree(ghostInterface.Ayz_h_fGhost.Y_1);
            cudaFree(ghostInterface.Ayz_h_fGhost.Z_0);
            cudaFree(ghostInterface.Ayz_h_fGhost.Z_1);
        #endif
        #ifdef A_ZZ_DIST
            cudaFree(ghostInterface.Azz_h_fGhost.X_0);
            cudaFree(ghostInterface.Azz_h_fGhost.X_1);
            cudaFree(ghostInterface.Azz_h_fGhost.Y_0);
            cudaFree(ghostInterface.Azz_h_fGhost.Y_1);
            cudaFree(ghostInterface.Azz_h_fGhost.Z_0);
            cudaFree(ghostInterface.Azz_h_fGhost.Z_1);
        #endif
/*  
        #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
            cudaFree(ghostInterface.h_f_uGhost.X_0);
            cudaFree(ghostInterface.h_f_uGhost.X_1);
            cudaFree(ghostInterface.h_f_uGhost.Y_0);
            cudaFree(ghostInterface.h_f_uGhost.Y_1);
            cudaFree(ghostInterface.h_f_uGhost.Z_0);
            cudaFree(ghostInterface.h_f_uGhost.Z_1);
        #endif


        #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
            cudaFree(ghostInterface.conf_h_fGhost.X_0);
            cudaFree(ghostInterface.conf_h_fGhost.X_1);
            cudaFree(ghostInterface.conf_h_fGhost.Y_0);
            cudaFree(ghostInterface.conf_h_fGhost.Y_1);
            cudaFree(ghostInterface.conf_h_fGhost.Z_0);
            cudaFree(ghostInterface.conf_h_fGhost.Z_1);
        #endif
    */
    }
}

/*
*   @brief Performs a CUDA memory copy for ghost interface data between source and destination.
*   @param ghostInterface: reference to the ghost interface data structure
*   @param dst: destination ghost data structure
*   @param src: source ghost data structure
*   @param kind: type of memory copy (e.g., cudaMemcpyHostToDevice)
*   @param Q: number of quantities in the ghost data that are transfered
*/
__host__
void interfaceCudaMemcpy(GhostInterfaceData& ghostInterface, ghostData& dst, const ghostData& src, cudaMemcpyKind kind, int Q) {
    struct MemcpyPair {
        dfloat* dst;
        const dfloat* src;
        size_t size;
    };

    MemcpyPair memcpyPairs[] = {
        { dst.X_0, src.X_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * Q},
        { dst.X_1, src.X_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * Q},
        { dst.Y_0, src.Y_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * Q},
        { dst.Y_1, src.Y_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * Q},
        { dst.Z_0, src.Z_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * Q},
        { dst.Z_1, src.Z_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * Q}
    };

    checkCudaErrors(cudaDeviceSynchronize());
    for (const auto& pair : memcpyPairs) {
        checkCudaErrors(cudaMemcpy(pair.dst, pair.src, pair.size, kind));
    }


}
/*
*   @brief Swaps the ghost interfaces.
*   @param ghostInterface: reference to the ghost interface data structure
*/
__host__
void swapGhostInterfaces(GhostInterfaceData& ghostInterface) {
    // Synchronize device before performing swaps
    checkCudaErrors(cudaDeviceSynchronize());

    // Swap interface pointers for fGhost and gGhost
    interfaceSwap(ghostInterface.fGhost.X_0, ghostInterface.gGhost.X_0);
    interfaceSwap(ghostInterface.fGhost.X_1, ghostInterface.gGhost.X_1);
    interfaceSwap(ghostInterface.fGhost.Y_0, ghostInterface.gGhost.Y_0);
    interfaceSwap(ghostInterface.fGhost.Y_1, ghostInterface.gGhost.Y_1);
    interfaceSwap(ghostInterface.fGhost.Z_0, ghostInterface.gGhost.Z_0);
    interfaceSwap(ghostInterface.fGhost.Z_1, ghostInterface.gGhost.Z_1);

    #ifdef SECOND_DIST
    // Swap interface pointers for g_fGhost and g_gGhost if SECOND_DIST is defined
    interfaceSwap(ghostInterface.g_fGhost.X_0, ghostInterface.g_gGhost.X_0);
    interfaceSwap(ghostInterface.g_fGhost.X_1, ghostInterface.g_gGhost.X_1);
    interfaceSwap(ghostInterface.g_fGhost.Y_0, ghostInterface.g_gGhost.Y_0);
    interfaceSwap(ghostInterface.g_fGhost.Y_1, ghostInterface.g_gGhost.Y_1);
    interfaceSwap(ghostInterface.g_fGhost.Z_0, ghostInterface.g_gGhost.Z_0);
    interfaceSwap(ghostInterface.g_fGhost.Z_1, ghostInterface.g_gGhost.Z_1);
    #endif

    #ifdef A_XX_DIST
    interfaceSwap(ghostInterface.Axx_fGhost.X_0, ghostInterface.Axx_gGhost.X_0);
    interfaceSwap(ghostInterface.Axx_fGhost.X_1, ghostInterface.Axx_gGhost.X_1);
    interfaceSwap(ghostInterface.Axx_fGhost.Y_0, ghostInterface.Axx_gGhost.Y_0);
    interfaceSwap(ghostInterface.Axx_fGhost.Y_1, ghostInterface.Axx_gGhost.Y_1);
    interfaceSwap(ghostInterface.Axx_fGhost.Z_0, ghostInterface.Axx_gGhost.Z_0);
    interfaceSwap(ghostInterface.Axx_fGhost.Z_1, ghostInterface.Axx_gGhost.Z_1);
    #endif

    #ifdef A_XY_DIST
    interfaceSwap(ghostInterface.Axy_fGhost.X_0, ghostInterface.Axy_gGhost.X_0);
    interfaceSwap(ghostInterface.Axy_fGhost.X_1, ghostInterface.Axy_gGhost.X_1);
    interfaceSwap(ghostInterface.Axy_fGhost.Y_0, ghostInterface.Axy_gGhost.Y_0);
    interfaceSwap(ghostInterface.Axy_fGhost.Y_1, ghostInterface.Axy_gGhost.Y_1);
    interfaceSwap(ghostInterface.Axy_fGhost.Z_0, ghostInterface.Axy_gGhost.Z_0);
    interfaceSwap(ghostInterface.Axy_fGhost.Z_1, ghostInterface.Axy_gGhost.Z_1);
    #endif

    #ifdef A_XZ_DIST
    interfaceSwap(ghostInterface.Axz_fGhost.X_0, ghostInterface.Axz_gGhost.X_0);
    interfaceSwap(ghostInterface.Axz_fGhost.X_1, ghostInterface.Axz_gGhost.X_1);
    interfaceSwap(ghostInterface.Axz_fGhost.Y_0, ghostInterface.Axz_gGhost.Y_0);
    interfaceSwap(ghostInterface.Axz_fGhost.Y_1, ghostInterface.Axz_gGhost.Y_1);
    interfaceSwap(ghostInterface.Axz_fGhost.Z_0, ghostInterface.Axz_gGhost.Z_0);
    interfaceSwap(ghostInterface.Axz_fGhost.Z_1, ghostInterface.Axz_gGhost.Z_1);
    #endif

    #ifdef A_YY_DIST
    interfaceSwap(ghostInterface.Ayy_fGhost.X_0, ghostInterface.Ayy_gGhost.X_0);
    interfaceSwap(ghostInterface.Ayy_fGhost.X_1, ghostInterface.Ayy_gGhost.X_1);
    interfaceSwap(ghostInterface.Ayy_fGhost.Y_0, ghostInterface.Ayy_gGhost.Y_0);
    interfaceSwap(ghostInterface.Ayy_fGhost.Y_1, ghostInterface.Ayy_gGhost.Y_1);
    interfaceSwap(ghostInterface.Ayy_fGhost.Z_0, ghostInterface.Ayy_gGhost.Z_0);
    interfaceSwap(ghostInterface.Ayy_fGhost.Z_1, ghostInterface.Ayy_gGhost.Z_1);
    #endif

    #ifdef A_YZ_DIST
    interfaceSwap(ghostInterface.Ayz_fGhost.X_0, ghostInterface.Ayz_gGhost.X_0);
    interfaceSwap(ghostInterface.Ayz_fGhost.X_1, ghostInterface.Ayz_gGhost.X_1);
    interfaceSwap(ghostInterface.Ayz_fGhost.Y_0, ghostInterface.Ayz_gGhost.Y_0);
    interfaceSwap(ghostInterface.Ayz_fGhost.Y_1, ghostInterface.Ayz_gGhost.Y_1);
    interfaceSwap(ghostInterface.Ayz_fGhost.Z_0, ghostInterface.Ayz_gGhost.Z_0);
    interfaceSwap(ghostInterface.Ayz_fGhost.Z_1, ghostInterface.Ayz_gGhost.Z_1);
    #endif

    #ifdef A_ZZ_DIST
    interfaceSwap(ghostInterface.Azz_fGhost.X_0, ghostInterface.Azz_gGhost.X_0);
    interfaceSwap(ghostInterface.Azz_fGhost.X_1, ghostInterface.Azz_gGhost.X_1);
    interfaceSwap(ghostInterface.Azz_fGhost.Y_0, ghostInterface.Azz_gGhost.Y_0);
    interfaceSwap(ghostInterface.Azz_fGhost.Y_1, ghostInterface.Azz_gGhost.Y_1);
    interfaceSwap(ghostInterface.Azz_fGhost.Z_0, ghostInterface.Azz_gGhost.Z_0);
    interfaceSwap(ghostInterface.Azz_fGhost.Z_1, ghostInterface.Azz_gGhost.Z_1);
    #endif
/*
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
        interfaceSwap(ghostInterface.f_uGhost.X_0, ghostInterface.g_uGhost.X_0);
        interfaceSwap(ghostInterface.f_uGhost.X_1, ghostInterface.g_uGhost.X_1);
        interfaceSwap(ghostInterface.f_uGhost.Y_0, ghostInterface.g_uGhost.Y_0);
        interfaceSwap(ghostInterface.f_uGhost.Y_1, ghostInterface.g_uGhost.Y_1);
        interfaceSwap(ghostInterface.f_uGhost.Z_0, ghostInterface.g_uGhost.Z_0);
        interfaceSwap(ghostInterface.f_uGhost.Z_1, ghostInterface.g_uGhost.Z_1);
    #endif

    #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
        interfaceSwap(ghostInterface.conf_fGhost.X_0, ghostInterface.conf_gGhost.X_0);
        interfaceSwap(ghostInterface.conf_fGhost.X_1, ghostInterface.conf_gGhost.X_1);
        interfaceSwap(ghostInterface.conf_fGhost.Y_0, ghostInterface.conf_gGhost.Y_0);
        interfaceSwap(ghostInterface.conf_fGhost.Y_1, ghostInterface.conf_gGhost.Y_1);
        interfaceSwap(ghostInterface.conf_fGhost.Z_0, ghostInterface.conf_gGhost.Z_0);
        interfaceSwap(ghostInterface.conf_fGhost.Z_1, ghostInterface.conf_gGhost.Z_1);
    #endif
    */
}


/*
*   @brief Allocates memory for the ghost interface data.
*   @param ghostInterface: reference to the ghost interface data structure
*/
__host__
void interfaceMalloc(ghostInterfaceData &ghostInterface)
{
    unsigned int memAllocated = 0;

    cudaMalloc((void **)&(ghostInterface.fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);
    cudaMalloc((void **)&(ghostInterface.fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);

    cudaMalloc((void **)&(ghostInterface.gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);
    cudaMalloc((void **)&(ghostInterface.gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);

    memAllocated = 2 * QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);

#ifdef SECOND_DIST
    cudaMalloc((void **)&(ghostInterface.g_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.g_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.g_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.g_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif

#ifdef A_XX_DIST
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axx_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif

#ifdef A_XY_DIST
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axy_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif

#ifdef A_XZ_DIST
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Axz_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif

#ifdef A_YY_DIST
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayy_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif

#ifdef A_YZ_DIST
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Ayz_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif

#ifdef A_ZZ_DIST
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);
    cudaMalloc((void **)&(ghostInterface.Azz_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    memAllocated += 2 * GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
#endif
/*
#ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
    cudaMalloc((void **)&(ghostInterface.f_uGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 3);
    cudaMalloc((void **)&(ghostInterface.f_uGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 3);
    cudaMalloc((void **)&(ghostInterface.f_uGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 3);
    cudaMalloc((void **)&(ghostInterface.f_uGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 3);
    cudaMalloc((void **)&(ghostInterface.f_uGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 3);
    cudaMalloc((void **)&(ghostInterface.f_uGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 3);

    cudaMalloc((void **)&(ghostInterface.g_uGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 3);
    cudaMalloc((void **)&(ghostInterface.g_uGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 3);
    cudaMalloc((void **)&(ghostInterface.g_uGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 3);
    cudaMalloc((void **)&(ghostInterface.g_uGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 3);
    cudaMalloc((void **)&(ghostInterface.g_uGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 3);
    cudaMalloc((void **)&(ghostInterface.g_uGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 3);
#endif


#ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
    cudaMalloc((void **)&(ghostInterface.conf_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 6);
    cudaMalloc((void **)&(ghostInterface.conf_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 6);

    cudaMalloc((void **)&(ghostInterface.conf_gGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_gGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_gGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_gGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 6);
    cudaMalloc((void **)&(ghostInterface.conf_gGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 6);
    cudaMalloc((void **)&(ghostInterface.conf_gGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 6);
#endif

*/

    if (LOAD_CHECKPOINT || CHECKPOINT_SAVE)
    {
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));

        memAllocated += QF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif

        #ifdef A_XX_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axx_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axy_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Axz_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayy_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Ayz_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.Azz_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));

        memAllocated += GF * (NUMBER_GHOST_FACE_YZ + NUMBER_GHOST_FACE_XZ + NUMBER_GHOST_FACE_XY) * sizeof(dfloat);
        #endif
/*
        #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_f_uGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 3));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_f_uGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 3));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_f_uGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 3));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_f_uGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 3));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_f_uGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 3));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_f_uGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 3));
        #endif

        
        #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.conf_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 6));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.conf_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * 6));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.conf_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 6));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.conf_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * 6));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.conf_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 6));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.conf_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * 6));
        #endif
        */
    }

    printf("Device Memory Allocated for Interface: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0)); if(console_flush) fflush(stdout);
}


__host__
void allocateHostMemory(
    dfloat** h_fMom, dfloat** rho, dfloat** ux, dfloat** uy, dfloat** uz
    OMEGA_FIELD_PARAMS_DECLARATION_PTR
    SECOND_DIST_PARAMS_DECLARATION_PTR
    A_XX_DIST_PARAMS_DECLARATION_PTR
    A_XY_DIST_PARAMS_DECLARATION_PTR
    A_XZ_DIST_PARAMS_DECLARATION_PTR
    A_YY_DIST_PARAMS_DECLARATION_PTR
    A_YZ_DIST_PARAMS_DECLARATION_PTR
    A_ZZ_DIST_PARAMS_DECLARATION_PTR
    MEAN_FLOW_PARAMS_DECLARATION_PTR
    MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR
    #if NODE_TYPE_SAVE
    ,unsigned int** nodeTypeSave
    #endif
    BC_FORCES_PARAMS_DECLARATION_PTR(h_)
) {
    unsigned int memAllocated = 0;

    checkCudaErrors(cudaMallocHost((void**)h_fMom, MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost((void**)rho, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)ux, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)uy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)uz, MEM_SIZE_SCALAR));

    memAllocated += MEM_SIZE_MOM + 4 * MEM_SIZE_SCALAR;

    #ifdef OMEGA_FIELD
    checkCudaErrors(cudaMallocHost((void**)omega, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif

    #ifdef SECOND_DIST
    checkCudaErrors(cudaMallocHost((void**)C, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif 

    #ifdef A_XX_DIST
    checkCudaErrors(cudaMallocHost((void**)Axx, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif 
    #ifdef A_XY_DIST
    checkCudaErrors(cudaMallocHost((void**)Axy, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif 
    #ifdef A_XZ_DIST
    checkCudaErrors(cudaMallocHost((void**)Axz, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif
    #ifdef A_YY_DIST
    checkCudaErrors(cudaMallocHost((void**)Ayy, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif
    #ifdef A_YZ_DIST
    checkCudaErrors(cudaMallocHost((void**)Ayz, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif 
    #ifdef A_ZZ_DIST
    checkCudaErrors(cudaMallocHost((void**)Azz, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif

    #ifdef LOG_CONFORMATION
        #ifdef A_XX_DIST
        checkCudaErrors(cudaMallocHost((void**)Cxx, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif 
        #ifdef A_XY_DIST
        checkCudaErrors(cudaMallocHost((void**)Cxy, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif 
        #ifdef A_XZ_DIST
        checkCudaErrors(cudaMallocHost((void**)Cxz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif
        #ifdef A_YY_DIST
        checkCudaErrors(cudaMallocHost((void**)Cyy, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif
        #ifdef A_YZ_DIST
        checkCudaErrors(cudaMallocHost((void**)Cyz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif 
        #ifdef A_ZZ_DIST
        checkCudaErrors(cudaMallocHost((void**)Czz, MEM_SIZE_SCALAR));
        memAllocated += MEM_SIZE_SCALAR;
        #endif
    #endif //LOG_CONFORMATION

    #if MEAN_FLOW
    checkCudaErrors(cudaMallocHost((void**)m_fMom, MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost((void**)m_rho, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)m_ux, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)m_uy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)m_uz, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_MOM + 4 * MEM_SIZE_SCALAR;
    #ifdef SECOND_DIST
    checkCudaErrors(cudaMallocHost((void**)m_c, MEM_SIZE_SCALAR));
    memAllocated += MEM_SIZE_SCALAR;
    #endif //SECOND_DIST
    #endif // MEAN_FLOW

    #ifdef BC_FORCES
    #ifdef SAVE_BC_FORCES
    checkCudaErrors(cudaMallocHost((void**)h_BC_Fx, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)h_BC_Fy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)h_BC_Fz, MEM_SIZE_SCALAR));
    memAllocated += 3 * MEM_SIZE_SCALAR;
    #endif
    #endif //_BC_FORCES

    
    #if NODE_TYPE_SAVE
    checkCudaErrors(cudaMallocHost((void**)nodeTypeSave, sizeof(unsigned int) * NUMBER_LBM_NODES));
    memAllocated += sizeof(unsigned int) * NUMBER_LBM_NODES;
    #endif //NODE_TYPE_SAVE

    printf("Host Memory Allocated: %0.2f MB\n", (float)memAllocated / (1024.0 * 1024.0)); if(console_flush) fflush(stdout);
}

__host__
void allocateDeviceMemory(
    dfloat** d_fMom, unsigned int** dNodeType, GhostInterfaceData* ghostInterface
    BC_FORCES_PARAMS_DECLARATION_PTR(d_)
) {
    unsigned int memAllocated = 0;

    cudaMalloc((void**)d_fMom, MEM_SIZE_MOM);
    cudaMalloc((void**)dNodeType, sizeof(int) * NUMBER_LBM_NODES);
    interfaceMalloc(*ghostInterface);

    memAllocated += MEM_SIZE_MOM + sizeof(int) * NUMBER_LBM_NODES;


    #ifdef BC_FORCES
    cudaMalloc((void**)d_BC_Fx, MEM_SIZE_SCALAR);
    cudaMalloc((void**)d_BC_Fy, MEM_SIZE_SCALAR);
    cudaMalloc((void**)d_BC_Fz, MEM_SIZE_SCALAR);
    memAllocated += 3 * MEM_SIZE_SCALAR;
    #endif

    printf("Device Memory Allocated for Bulk flow: %.2f MB \n", (float)memAllocated /(1024.0 * 1024.0));
}


__host__
void initializeDomain(
    GhostInterfaceData &ghostInterface, 
    dfloat *&d_fMom, dfloat *&h_fMom, 
    #if MEAN_FLOW
    dfloat *&m_fMom, 
    #endif
    unsigned int *&hNodeType, unsigned int *&dNodeType, dfloat **&randomNumbers,
    BC_FORCES_PARAMS_DECLARATION(&d_)
    DENSITY_CORRECTION_PARAMS_DECLARATION(&h_)
    DENSITY_CORRECTION_PARAMS_DECLARATION(&d_)
    int *step, dim3 gridBlock, dim3 threadBlock
    ){
    
    // Random numbers initialization
    #ifdef RANDOM_NUMBERS 
        if(console_flush) fflush(stdout);
        checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[0], sizeof(float) * NUMBER_LBM_NODES));
        initializationRandomNumbers(randomNumbers[0], CURAND_SEED);
        checkCudaErrors(cudaDeviceSynchronize());
        getLastCudaError("random numbers transfer error");
        printf("Random numbers initialized - Seed used: %u\n", CURAND_SEED); 
        printf("Device memory allocated for random numbers: %.2f MB\n", (float)(sizeof(float) * NUMBER_LBM_NODES) / (1024.0 * 1024.0));
        if(console_flush) fflush(stdout);
    #endif //RANDOM_NUMBERS

    int checkpoint_state = 0;
    // LBM Initialization
    if (LOAD_CHECKPOINT) {

        printf("Loading checkpoint\n");
        checkpoint_state = loadSimCheckpoint(h_fMom, ghostInterface, step);

        if (checkpoint_state != 0){
            checkCudaErrors(cudaMemcpy(d_fMom, h_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyHostToDevice));
            interfaceCudaMemcpy(ghostInterface, ghostInterface.fGhost, ghostInterface.h_fGhost, cudaMemcpyHostToDevice, QF);

            #ifdef SECOND_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.g_fGhost, ghostInterface.g_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif

            #ifdef A_XX_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.Axx_fGhost, ghostInterface.Axx_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif
            #ifdef A_XY_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.Axy_fGhost, ghostInterface.Axy_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif
            #ifdef A_XZ_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.Axz_fGhost, ghostInterface.Axz_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif
            #ifdef A_YY_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayy_fGhost, ghostInterface.Ayy_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif
            #ifdef A_YZ_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayz_fGhost, ghostInterface.Ayz_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif
            #ifdef A_ZZ_DIST
                interfaceCudaMemcpy(ghostInterface, ghostInterface.Azz_fGhost, ghostInterface.Azz_h_fGhost, cudaMemcpyHostToDevice, GF);
            #endif
            /*
            #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
                interfaceCudaMemcpy(ghostInterface, ghostInterface.f_uGhost, ghostInterface.h_f_uGhost, cudaMemcpyHostToDevice, 3);
            #endif
    
            
            #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
                interfaceCudaMemcpy(ghostInterface, ghostInterface.conf_fGhost, ghostInterface.conf_h_fGhost, cudaMemcpyHostToDevice, 6);
            #endif
            */
        }
    } 
    if (!checkpoint_state) {
        if (LOAD_FIELD) {
            // Implement LOAD_FIELD logic if needed
        } else {
            gpuInitialization_mom<<<gridBlock, threadBlock>>>(d_fMom, randomNumbers[0]);
        }
        gpuInitialization_pop<<<gridBlock, threadBlock>>>(d_fMom, ghostInterface);
    }

    // Mean flow initialization
    #if MEAN_FLOW
        checkCudaErrors(cudaMemcpy(m_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToDevice));
    #endif

    // Node type initialization
    checkCudaErrors(cudaMallocHost((void**)&hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES));
    #if NODE_TYPE_SAVE
        checkCudaErrors(cudaMallocHost((void**)&dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES));
    #endif

    #ifndef VOXEL_FILENAME
        hostInitialization_nodeType(hNodeType);
        checkCudaErrors(cudaMemcpy(dNodeType, hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
        checkCudaErrors(cudaDeviceSynchronize());
    #else
        hostInitialization_nodeType_bulk(hNodeType); 
        read_xyz_file(VOXEL_FILENAME, hNodeType);
        hostInitialization_nodeType(hNodeType);
        checkCudaErrors(cudaMemcpy(dNodeType, hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
        checkCudaErrors(cudaDeviceSynchronize());
        define_voxel_bc<<<gridBlock, threadBlock>>>(dNodeType); 
        checkCudaErrors(cudaMemcpy(hNodeType, dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyDeviceToHost)); 
    #endif

    // Boundary condition forces initialization
    #ifdef BC_FORCES
        gpuInitialization_force<<<gridBlock, threadBlock>>>(d_BC_Fx, d_BC_Fy, d_BC_Fz);
    #endif

    // Interface population initialization
    interfaceCudaMemcpy(ghostInterface, ghostInterface.gGhost, ghostInterface.fGhost, cudaMemcpyDeviceToDevice, QF);
    #ifdef SECOND_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.g_gGhost, ghostInterface.g_fGhost, cudaMemcpyDeviceToDevice, GF);
        printf("Interface pop copied \n"); if(console_flush) fflush(stdout);
    #endif
    #ifdef A_XX_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.Axx_gGhost, ghostInterface.Axx_fGhost, cudaMemcpyDeviceToDevice, GF);
    #endif
    #ifdef A_XY_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.Axy_gGhost, ghostInterface.Axy_fGhost, cudaMemcpyDeviceToDevice, GF);
    #endif
    #ifdef A_XZ_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.Axz_gGhost, ghostInterface.Axz_fGhost, cudaMemcpyDeviceToDevice, GF);
    #endif
    #ifdef A_YY_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayy_gGhost, ghostInterface.Ayy_fGhost, cudaMemcpyDeviceToDevice, GF);
    #endif
    #ifdef A_YZ_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.Ayz_gGhost, ghostInterface.Ayz_fGhost, cudaMemcpyDeviceToDevice, GF);
    #endif
    #ifdef A_ZZ_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.Azz_gGhost, ghostInterface.Azz_fGhost, cudaMemcpyDeviceToDevice, GF);
    #endif
    
    /*
    #ifdef COMPUTE_VEL_GRADIENT_FINITE_DIFFERENCE
        interfaceCudaMemcpy(ghostInterface, ghostInterface.f_uGhost, ghostInterface.g_uGhost, cudaMemcpyDeviceToDevice, 3);
    #endif
    #ifdef COMPUTE_CONF_GRADIENT_FINITE_DIFFERENCE
        interfaceCudaMemcpy(ghostInterface, ghostInterface.conf_fGhost, ghostInterface.conf_gGhost, cudaMemcpyDeviceToDevice, 3);
    #endif
    */

    // Synchronize after all initializations
    checkCudaErrors(cudaDeviceSynchronize());


    // Synchronize and transfer data back to host if needed
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Synchorizing data back to host \n"); if(console_flush) fflush(stdout);

    // Free random numbers if initialized
    #ifdef RANDOM_NUMBERS
        checkCudaErrors(cudaSetDevice(GPU_INDEX));
        cudaFree(randomNumbers[0]);
        free(randomNumbers);
        printf("Random numbers free \n"); if(console_flush) fflush(stdout);
    #endif
}


#ifdef PARTICLE_MODEL
__host__
void initializeParticle(ParticlesSoA& particlesSoA, Particle *particles, int *step, dim3 gridBlock, dim3 threadBlock){

    printf("Creating particles...\t"); fflush(stdout);
    particlesSoA.createParticles(particles);
    printf("Particles created!\n"); fflush(stdout);

    particlesSoA.updateParticlesAsSoA(particles);
    printf("Update ParticlesAsSoA!\n"); fflush(stdout);

    int checkpoint_state = 0;
    // Checar se exite checkpoint
    if(LOAD_CHECKPOINT)
    {
        checkpoint_state = loadSimCheckpointParticle(particlesSoA, step);
       
    }else{
        if(checkpoint_state != 0){
            step = INI_STEP;
            dim3 gridInit = gridBlock;
            // Initialize ghost nodes
            gridInit.z += 1;
            
            checkCudaErrors(cudaSetDevice(GPU_INDEX));
            // Initialize populations
            //gpuInitialization<<<gridInit, threads>>>(pop[i], macr[i], randomNumbers[i]);
            checkCudaErrors(cudaDeviceSynchronize());

            getLastCudaError("Initialization error");
        }
    }

}
#endif // PARTICLE_MODEL
#endif // MAIN_CUH
