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
#ifdef NON_NEWTONIAN_FLUID
    #include "nnf.h"
#endif
#ifdef PARTICLE_TRACER
    #include "particleTracer.cuh"
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

    if (LOAD_CHECKPOINT)
    {
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
*   @brief Allocates memory for the ghost interface data.
*   @param ghostInterface: reference to the ghost interface data structure
*/
__host__
void interfaceMalloc(ghostInterfaceData &ghostInterface)
{
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
#endif

    if (LOAD_CHECKPOINT || CHECKPOINT_SAVE)
    {
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));

#ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.X_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.X_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Y_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Y_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Z_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void **)&(ghostInterface.g_h_fGhost.Z_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
#endif
    }
}

/*
*   @brief Swaps the ghost interfaces.
*   @param ghostInterface: reference to the ghost interface data structure
*/
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
}

#endif // MAIN_CUH
