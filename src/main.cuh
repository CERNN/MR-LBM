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

void initializeCudaEvents(cudaEvent_t &start, cudaEvent_t &stop, cudaEvent_t &start_step, cudaEvent_t &stop_step) {
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));
}


dfloat recordElapsedTime(cudaEvent_t &start_step, cudaEvent_t &stop_step, int step) {
    checkCudaErrors(cudaEventRecord(stop_step, 0));
    checkCudaErrors(cudaEventSynchronize(stop_step));
    
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start_step, stop_step));
    elapsedTime *= 0.001;

    size_t nodesUpdatedSync = step * NUMBER_LBM_NODES;
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


__host__
void allocateHostMemory(
    dfloat** h_fMom, dfloat** rho, dfloat** ux, dfloat** uy, dfloat** uz
    NON_NEWTONIAN_FLUID_PARAMS_DECLARATION_PTR
    SECOND_DIST_PARAMS_DECLARATION_PTR
    PARTICLE_TRACER_PARAMS_DECLARATION_PTR(h_)
    MEAN_FLOW_PARAMS_DECLARATION_PTR
    MEAN_FLOW_SECOND_DIST_PARAMS_DECLARATION_PTR
    BC_FORCES_PARAMS_DECLARATION_PTR(h_)
) {
    checkCudaErrors(cudaMallocHost((void**)h_fMom, MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost((void**)rho, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)ux, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)uy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)uz, MEM_SIZE_SCALAR));

    #ifdef NON_NEWTONIAN_FLUID
    checkCudaErrors(cudaMallocHost((void**)omega, MEM_SIZE_SCALAR));
    #endif

    #ifdef SECOND_DIST
    checkCudaErrors(cudaMallocHost((void**)C, MEM_SIZE_SCALAR));
    #endif 

    #ifdef PARTICLE_TRACER
    checkCudaErrors(cudaMallocHost((void**)h_particlePos, sizeof(dfloat3) * NUM_PARTICLES));
    #endif

    #if MEAN_FLOW
    checkCudaErrors(cudaMallocHost((void**)m_fMom, MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost((void**)m_rho, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)m_ux, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)m_uy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)m_uz, MEM_SIZE_SCALAR));
    #ifdef SECOND_DIST
    checkCudaErrors(cudaMallocHost((void**)m_c, MEM_SIZE_SCALAR));
    #endif
    #endif // MEAN_FLOW

    #ifdef BC_FORCES
    #ifdef SAVE_BC_FORCES
    checkCudaErrors(cudaMallocHost((void**)h_BC_Fx, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)h_BC_Fy, MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)h_BC_Fz, MEM_SIZE_SCALAR));
    #endif
    #endif //_BC_FORCES
}

__host__
void allocateDeviceMemory(
    dfloat** d_fMom, unsigned int** dNodeType, GhostInterfaceData* ghostInterface
    PARTICLE_TRACER_PARAMS_DECLARATION_PTR(d_)
    BC_FORCES_PARAMS_DECLARATION_PTR(d_)
) {
    cudaMalloc((void**)d_fMom, MEM_SIZE_MOM);
    cudaMalloc((void**)dNodeType, sizeof(int) * NUMBER_LBM_NODES);
    interfaceMalloc(*ghostInterface);

    #ifdef PARTICLE_TRACER
    checkCudaErrors(cudaMalloc((void**)d_particlePos, sizeof(dfloat3) * NUM_PARTICLES));
    #endif

    #ifdef BC_FORCES
    cudaMalloc((void**)d_BC_Fx, MEM_SIZE_SCALAR);
    cudaMalloc((void**)d_BC_Fy, MEM_SIZE_SCALAR);
    cudaMalloc((void**)d_BC_Fz, MEM_SIZE_SCALAR);
    #endif
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
    PARTICLE_TRACER_PARAMS_DECLARATION(&h_)
    PARTICLE_TRACER_PARAMS_DECLARATION(&d_)
    int *step, dim3 gridBlock, dim3 threadBlock
    ){
    
    // Random numbers initialization
    #ifdef RANDOM_NUMBERS
        printf("Initializing random numbers\n"); 
        if(console_flush) fflush(stdout);
        checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[0], sizeof(float) * NUMBER_LBM_NODES));
        initializationRandomNumbers(randomNumbers[0], CURAND_SEED);
        checkCudaErrors(cudaDeviceSynchronize());
        getLastCudaError("random numbers transfer error");
        printf("Random numbers initialized\n");
        if(console_flush) fflush(stdout);
    #endif

    // LBM Initialization
    if (LOAD_CHECKPOINT) {
        printf("Loading checkpoint\n");
        step[0] = INI_STEP;
        loadSimCheckpoint(h_fMom, ghostInterface, step);
        checkCudaErrors(cudaMemcpy(d_fMom, h_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyHostToDevice));
        interfaceCudaMemcpy(ghostInterface, ghostInterface.fGhost, ghostInterface.h_fGhost, cudaMemcpyHostToDevice, QF);
        #ifdef SECOND_DIST
            interfaceCudaMemcpy(ghostInterface, ghostInterface.g_fGhost, ghostInterface.g_h_fGhost, cudaMemcpyHostToDevice, GF);
        #endif
    } else {
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
        checkCudaErrors(cudaMallocHost((void**)&nodeTypeSave, sizeof(dfloat) * NUMBER_LBM_NODES));
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
    #ifdef SECOND_DIST
        interfaceCudaMemcpy(ghostInterface, ghostInterface.gGhost, ghostInterface.fGhost, cudaMemcpyDeviceToDevice, QF);
        interfaceCudaMemcpy(ghostInterface, ghostInterface.g_gGhost, ghostInterface.g_fGhost, cudaMemcpyDeviceToDevice, GF);
        printf("Interface pop copied \n"); if(console_flush) fflush(stdout);
    #endif


    // Synchronize after all initializations
    checkCudaErrors(cudaDeviceSynchronize());

    // Particle tracer initialization
    #ifdef PARTICLE_TRACER
        initializeParticles(h_particlePos, d_particlePos);
    #endif

    // Synchronize and transfer data back to host if needed
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES * NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    printf("Mom copy to host \n"); if(console_flush) fflush(stdout);

    // Free random numbers if initialized
    #ifdef RANDOM_NUMBERS
        checkCudaErrors(cudaSetDevice(GPU_INDEX));
        cudaFree(randomNumbers[0]);
        free(randomNumbers);
        printf("Random numbers free \n"); if(console_flush) fflush(stdout);
    #endif
}


#endif // MAIN_CUH
