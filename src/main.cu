#include <stdio.h>
#include <stdlib.h>

// CUDA INCLUDE
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// FILES INCLUDES
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

using namespace std;

__host__ __device__
void interfaceSwap(dfloat* &pt1, dfloat* &pt2){
  dfloat *temp = pt1;
  pt1 = pt2;
  pt2 = temp;
} 

int main() {
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    dfloat* fMom;
    dfloat* fGhostX_0;
    dfloat* fGhostX_1;
    dfloat* fGhostY_0; 
    dfloat* fGhostY_1;
    dfloat* fGhostZ_0; 
    dfloat* fGhostZ_1;

    dfloat* gGhostX_0;
    dfloat* gGhostX_1;
    dfloat* gGhostY_0; 
    dfloat* gGhostY_1;
    dfloat* gGhostZ_0; 
    dfloat* gGhostZ_1;

    dfloat* h_fGhostX_0;
    dfloat* h_fGhostX_1;
    dfloat* h_fGhostY_0; 
    dfloat* h_fGhostY_1;
    dfloat* h_fGhostZ_0; 
    dfloat* h_fGhostZ_1;

    unsigned int* dNodeType;
    unsigned int* hNodeType;
    #if SAVE_BC
    dfloat* nodeTypeSave;
    #endif

    #ifdef DENSITY_CORRECTION
    dfloat* h_mean_rho;
    dfloat* d_mean_rho;
    #endif

    #ifdef PARTICLE_TRACER
    dfloat3* h_particlePos;
    dfloat3* d_particlePos;
    #endif

    #if MEAN_FLOW
        dfloat* m_fMom;
        dfloat* m_rho;
        dfloat* m_ux;
        dfloat* m_uy;
        dfloat* m_uz;
        #ifdef SECOND_DIST
        dfloat* m_c;
        #endif
    #endif //MEAN_FLOW

    #ifdef BC_FORCES
        #ifdef SAVE_BC_FORCES
        dfloat* h_BC_Fx;
        dfloat* h_BC_Fy;
        dfloat* h_BC_Fz;
        #endif

        dfloat* d_BC_Fx;
        dfloat* d_BC_Fy;
        dfloat* d_BC_Fz;
    #endif //_BC_FORCES

    #ifdef SECOND_DIST
        dfloat* g_fGhostX_0;
        dfloat* g_fGhostX_1;
        dfloat* g_fGhostY_0; 
        dfloat* g_fGhostY_1;
        dfloat* g_fGhostZ_0; 
        dfloat* g_fGhostZ_1;

        dfloat* g_gGhostX_0;
        dfloat* g_gGhostX_1;
        dfloat* g_gGhostY_0; 
        dfloat* g_gGhostY_1;
        dfloat* g_gGhostZ_0; 
        dfloat* g_gGhostZ_1;

        dfloat* g_h_fGhostX_0;
        dfloat* g_h_fGhostX_1;
        dfloat* g_h_fGhostY_0; 
        dfloat* g_h_fGhostY_1;
        dfloat* g_h_fGhostZ_0; 
        dfloat* g_h_fGhostZ_1;
    #endif




    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    dfloat* h_fMom;
    dfloat* rho;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;

    int step = 0;

    #ifdef NON_NEWTONIAN_FLUID
    dfloat* omega;
    #endif

    #ifdef SECOND_DIST
    dfloat* C;
    #endif 


    float** randomNumbers = nullptr; // useful for turbulence

    checkCudaErrors(cudaMallocHost((void**)&(h_fMom), MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost((void**)&(rho), MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)&(ux), MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)&(uy), MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)&(uz), MEM_SIZE_SCALAR));
    #ifdef NON_NEWTONIAN_FLUID
    checkCudaErrors(cudaMallocHost((void**)&(omega), MEM_SIZE_SCALAR));
    #endif
    #ifdef SECOND_DIST
    checkCudaErrors(cudaMallocHost((void**)&(C), MEM_SIZE_SCALAR));
    #endif 
    #ifdef PARTICLE_TRACER
    checkCudaErrors(cudaMallocHost((void**)&(h_particlePos), sizeof(dfloat3)*NUM_PARTICLES));
    #endif
    #if MEAN_FLOW
        checkCudaErrors(cudaMallocHost((void**)&(m_fMom), MEM_SIZE_MOM));
        checkCudaErrors(cudaMallocHost((void**)&(m_rho), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(m_ux), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(m_uy), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(m_uz), MEM_SIZE_SCALAR));
        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void**)&(m_c), MEM_SIZE_SCALAR));
        #endif
    #endif //MEAN_FLOW
    #ifdef BC_FORCES
        #ifdef SAVE_BC_FORCES
        checkCudaErrors(cudaMallocHost((void**)&(h_BC_Fx), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(h_BC_Fy), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(h_BC_Fz), MEM_SIZE_SCALAR));
        #endif
    #endif //_BC_FORCES
    randomNumbers = (float**)malloc(sizeof(float*));


    // Setup saving folder
    folderSetup();

    /* -------------- ALLOCATION AND CONFIGURATION FOR EACH GPU ------------- */

    cudaMalloc((void**)&fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS);  
    cudaMalloc((void**)&dNodeType, sizeof(int) * NUMBER_LBM_NODES);  

    cudaMalloc((void**)&fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);    
    cudaMalloc((void**)&fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);    
    cudaMalloc((void**)&fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);    
    cudaMalloc((void**)&fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);    
    cudaMalloc((void**)&fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);    
    cudaMalloc((void**)&fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);

    cudaMalloc((void**)&gGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);    
    cudaMalloc((void**)&gGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);    
    cudaMalloc((void**)&gGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);    
    cudaMalloc((void**)&gGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);    
    cudaMalloc((void**)&gGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);    
    cudaMalloc((void**)&gGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);    

    #ifdef SECOND_DIST
    cudaMalloc((void**)&g_fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);    
    cudaMalloc((void**)&g_fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);    
    cudaMalloc((void**)&g_fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);    
    cudaMalloc((void**)&g_fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);    
    cudaMalloc((void**)&g_fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);    
    cudaMalloc((void**)&g_fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);

    cudaMalloc((void**)&g_gGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);    
    cudaMalloc((void**)&g_gGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF);    
    cudaMalloc((void**)&g_gGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);    
    cudaMalloc((void**)&g_gGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF);    
    cudaMalloc((void**)&g_gGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);    
    cudaMalloc((void**)&g_gGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF);    
    #endif 
     

    #ifdef DENSITY_CORRECTION
        checkCudaErrors(cudaMallocHost((void**)&(h_mean_rho), sizeof(dfloat)));
        cudaMalloc((void**)&d_mean_rho, sizeof(dfloat));  
    #endif
    #ifdef PARTICLE_TRACER
    checkCudaErrors(cudaMalloc((void**)&(d_particlePos), sizeof(dfloat3)*NUM_PARTICLES));
    #endif

    #ifdef BC_FORCES
        cudaMalloc((void**)&d_BC_Fx, MEM_SIZE_SCALAR);    
        cudaMalloc((void**)&d_BC_Fy, MEM_SIZE_SCALAR);    
        cudaMalloc((void**)&d_BC_Fz, MEM_SIZE_SCALAR);            
    #endif //_BC_FORCES
    //printf("Allocated memory \n"); if(console_flush){fflush(stdout);}
    

    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef PARTICLE_TRACER
    cudaStream_t streamsPart[1];
    checkCudaErrors(cudaStreamCreate(&streamsPart[0]));
    #endif

    if(RANDOM_NUMBERS)
    {   
        //printf("Initializing random numbers\n");if(console_flush){fflush(stdout);}
        checkCudaErrors(cudaMallocManaged((void**)&randomNumbers[0], 
            sizeof(float)*NUMBER_LBM_NODES));
        initializationRandomNumbers(randomNumbers[0], CURAND_SEED);
        checkCudaErrors(cudaDeviceSynchronize());
        getLastCudaError("random numbers transfer error");
        //printf("random numbers initialized \n");if(console_flush){fflush(stdout);}
    }

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- LBM INITIALIZATION ------------------------- */
    if(LOAD_CHECKPOINT || CHECKPOINT_SAVE){
        checkCudaErrors(cudaMallocHost((void**)&(h_fGhostX_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void**)&(h_fGhostX_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF));
        checkCudaErrors(cudaMallocHost((void**)&(h_fGhostY_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void**)&(h_fGhostY_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF));
        checkCudaErrors(cudaMallocHost((void**)&(h_fGhostZ_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));
        checkCudaErrors(cudaMallocHost((void**)&(h_fGhostZ_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF));

        #ifdef SECOND_DIST
        checkCudaErrors(cudaMallocHost((void**)&(g_h_fGhostX_0), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void**)&(g_h_fGhostX_1), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF));
        checkCudaErrors(cudaMallocHost((void**)&(g_h_fGhostY_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void**)&(g_h_fGhostY_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF));
        checkCudaErrors(cudaMallocHost((void**)&(g_h_fGhostZ_0), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));
        checkCudaErrors(cudaMallocHost((void**)&(g_h_fGhostZ_1), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF));  
        #endif 
    }
    if(LOAD_CHECKPOINT){
        step = INI_STEP;
        loadSimCheckpoint(h_fMom, h_fGhostX_0,h_fGhostX_1,h_fGhostY_0,h_fGhostY_1,h_fGhostZ_0,h_fGhostZ_1,
        #ifdef SECOND_DIST 
        g_fGhostX_0, g_fGhostX_1,g_fGhostY_0, g_fGhostY_1, g_fGhostZ_0, g_fGhostZ_1,
        #endif 
        &step);

        checkCudaErrors(cudaMemcpy(fMom, h_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(fGhostX_0, h_fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(fGhostX_1, h_fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(fGhostY_0, h_fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(fGhostY_1, h_fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(fGhostZ_0, h_fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(fGhostZ_1, h_fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyHostToDevice));
       
        #ifdef SECOND_DIST 
        checkCudaErrors(cudaMemcpy(g_fGhostX_0, g_h_fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(g_fGhostX_1, g_h_fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(g_fGhostY_0, g_h_fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(g_fGhostY_1, g_h_fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(g_fGhostZ_0, g_h_fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(g_fGhostZ_1, g_h_fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, cudaMemcpyHostToDevice));
        #endif 
       

    }else{
        gpuInitialization_mom << <gridBlock, threadBlock >> >(fMom, randomNumbers[0]);
        //printf("Moments initialized \n");if(console_flush){fflush(stdout);}
        gpuInitialization_pop << <gridBlock, threadBlock >> >(fMom,fGhostX_0,fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1
        #ifdef SECOND_DIST 
        ,g_fGhostX_0, g_fGhostX_1, g_fGhostY_0, g_fGhostY_1,g_fGhostZ_0, g_fGhostZ_1
        #endif 
        );
    }

    #if MEAN_FLOW
        //initialize mean moments
        checkCudaErrors(cudaMemcpy(m_fMom,fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToDevice));
    #endif //MEAN_FLOW
    checkCudaErrors(cudaMallocHost((void**)&(hNodeType), sizeof(unsigned int) * NUMBER_LBM_NODES));
    #if SAVE_BC
    checkCudaErrors(cudaMallocHost((void**)&(nodeTypeSave), sizeof(dfloat) * NUMBER_LBM_NODES));
    #endif 

    #ifndef VOXEL_FILENAME
    //gpuInitialization_nodeType << <gridBlock, threadBlock >> >(dNodeType);
    //checkCudaErrors(cudaDeviceSynchronize());
        hostInitialization_nodeType(hNodeType);
        checkCudaErrors(cudaMemcpy(dNodeType, hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  
        checkCudaErrors(cudaDeviceSynchronize());
    #endif
    #ifdef VOXEL_FILENAME
        hostInitialization_nodeType_bulk(hNodeType); //initialize the domain with  BULK
        read_xyz_file(VOXEL_FILENAME,hNodeType); //overwrite the domain with the voxels information + add missing defintion 
        hostInitialization_nodeType(hNodeType); //initialize the domain with BC
        checkCudaErrors(cudaMemcpy(dNodeType, hNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyHostToDevice));  // copy inform\ation to device
        checkCudaErrors(cudaDeviceSynchronize());
        define_voxel_bc << <gridBlock, threadBlock >> >(dNodeType); //update information of BC condition nearby the voxels
        checkCudaErrors(cudaMemcpy(hNodeType, dNodeType, sizeof(unsigned int) * NUMBER_LBM_NODES, cudaMemcpyDeviceToHost)); 
    #endif

    #ifdef BC_FORCES
    gpuInitialization_force << <gridBlock, threadBlock >> >(d_BC_Fx,d_BC_Fy,d_BC_Fz);
    #endif //_BC_FORCES

    //printf("Interface Populations initialized \n");if(console_flush){fflush(stdout);}
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(gGhostX_0, fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostX_1, fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostY_0, fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostY_1, fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostZ_0, fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostZ_1, fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToDevice));

    #ifdef SECOND_DIST 
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(g_gGhostX_0, g_fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(g_gGhostX_1, g_fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(g_gGhostY_0, g_fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(g_gGhostY_1, g_fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(g_gGhostZ_0, g_fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(g_gGhostZ_1, g_fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, cudaMemcpyDeviceToDevice));
    #endif 
    #ifdef DENSITY_CORRECTION
        h_mean_rho[0] = RHO_0;
        checkCudaErrors(cudaMemcpy(d_mean_rho, h_mean_rho, sizeof(dfloat), cudaMemcpyHostToDevice)); 
    #endif
    checkCudaErrors(cudaDeviceSynchronize());

    #ifdef PARTICLE_TRACER
        initializeParticles(h_particlePos,d_particlePos);
    #endif

    
    //printf("step %zu\t",step); if(console_flush){fflush(stdout);}


    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    linearMacr(h_fMom,rho,ux,uy,uz,
    #ifdef NON_NEWTONIAN_FLUID
    omega,
    #endif
    #ifdef SECOND_DIST 
    C,
    #endif 
    #if SAVE_BC
    nodeTypeSave,
    hNodeType,
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    h_BC_Fx,
    h_BC_Fy,
    h_BC_Fz,
    #endif
    step); 

    // Free random numbers
    if (RANDOM_NUMBERS) {
        checkCudaErrors(cudaSetDevice(GPU_INDEX));
        cudaFree(randomNumbers[0]);
        free(randomNumbers);
    }

   
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    cudaEvent_t start, stop, start_step, stop_step;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));
    /* ------------------------------ LBM LOOP ------------------------------ */
    #ifdef DYNAMIC_SHARED_MEMORY
    cudaFuncSetAttribute(gpuMomCollisionStream, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEMORY_SIZE); // DOESNT WORK: DYNAMICALLY SHARED MEMORY HAS WORSE PERFORMANCE
    #endif
    for (step=1; step<N_STEPS;step++){

        int aux = step-INI_STEP;
        bool checkpoint = false;
        #ifdef DENSITY_CORRECTION
        mean_rho(fMom,step,d_mean_rho);
        #endif 
        bool save =false;
        if(aux != 0){
            if(MACR_SAVE)
                save = !(step % MACR_SAVE);
            if(CHECKPOINT_SAVE)
                checkpoint = !(aux % CHECKPOINT_SAVE);
        }
       



        gpuMomCollisionStream << <gridBlock, threadBlock 
        #ifdef DYNAMIC_SHARED_MEMORY
        , SHARED_MEMORY_SIZE
        #endif
        >> > (fMom,dNodeType,
        fGhostX_0,fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1,
        gGhostX_0,gGhostX_1,gGhostY_0,gGhostY_1,gGhostZ_0,gGhostZ_1,
        #ifdef SECOND_DIST 
        g_fGhostX_0,g_fGhostX_1,g_fGhostY_0,g_fGhostY_1,g_fGhostZ_0,g_fGhostZ_1,
        g_gGhostX_0,g_gGhostX_1,g_gGhostY_0,g_gGhostY_1,g_gGhostZ_0,g_gGhostZ_1,
        #endif 
        #ifdef DENSITY_CORRECTION
        d_mean_rho,
        #endif
        #ifdef BC_FORCES
        d_BC_Fx,d_BC_Fy,d_BC_Fz,
        #endif 
        step,
        save); 

        #ifdef PARTICLE_TRACER
            checkCudaErrors(cudaDeviceSynchronize());
            updateParticlePos(d_particlePos, h_particlePos, fMom, streamsPart[0],step);
        #endif

        if(checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
            checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_fGhostX_0,gGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_fGhostX_1,gGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_fGhostY_0,gGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_fGhostY_1,gGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_fGhostZ_0,gGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_fGhostZ_1,gGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaDeviceSynchronize());
           
            saveSimCheckpoint(fMom,gGhostX_0,gGhostX_1,gGhostY_0,gGhostY_1,gGhostZ_0,gGhostZ_1,
            #ifdef SECOND_DIST 
            g_fGhostX_0, g_fGhostX_1, g_fGhostY_0, g_fGhostY_1,g_fGhostZ_0, g_fGhostZ_1,
            #endif 
            &step);
        }


        
        //swap interface pointers
        checkCudaErrors(cudaDeviceSynchronize());
        interfaceSwap(fGhostX_0,gGhostX_0);
        interfaceSwap(fGhostX_1,gGhostX_1);
        interfaceSwap(fGhostY_0,gGhostY_0);
        interfaceSwap(fGhostY_1,gGhostY_1);
        interfaceSwap(fGhostZ_0,gGhostZ_0);
        interfaceSwap(fGhostZ_1,gGhostZ_1);

        #ifdef SECOND_DIST
        interfaceSwap(g_fGhostX_0,g_gGhostX_0);
        interfaceSwap(g_fGhostX_1,g_gGhostX_1);
        interfaceSwap(g_fGhostY_0,g_gGhostY_0);
        interfaceSwap(g_fGhostY_1,g_gGhostY_1);
        interfaceSwap(g_fGhostZ_0,g_gGhostZ_0);
        interfaceSwap(g_fGhostZ_1,g_gGhostZ_1);
        #endif
        

        //save macroscopics

        if(save){
            //if (N_STEPS - step < 4*((int)turn_over_time)){

            #if TREATFIELD
            treatData(h_fMom,fMom,
            #if MEAN_FLOW
            m_fMom,
            #endif //MEAN_FLOW
            step);
            //totalKineticEnergy(fMom,step);
            #endif //TREATFIELD
         
            #if TREATPOINT
                probeExport(fMom,
                #ifdef NON_NEWTONIAN_FLUID
                omega,
                #endif
                step);
            #endif
            #if TREATLINE
            velocityProfile(fMom,1,step);
            velocityProfile(fMom,2,step);
            velocityProfile(fMom,3,step);
            velocityProfile(fMom,4,step);
            velocityProfile(fMom,5,step);
            velocityProfile(fMom,6,step);
            #endif
            
            #if defined BC_FORCES && defined SAVE_BC_FORCES
            checkCudaErrors(cudaDeviceSynchronize()); 
            checkCudaErrors(cudaMemcpy(h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            #endif
            //if (!(step%((int)turn_over_time/10))){
            //if((step>N_STEPS-25*(int)(turn_over_time))){ 
            //    if((step%((int)(turn_over_time/2))) == 0){
                    checkCudaErrors(cudaDeviceSynchronize()); 
                    checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
                    linearMacr(h_fMom,rho,ux,uy,uz,
                    #ifdef NON_NEWTONIAN_FLUID
                    omega,
                    #endif
                    #ifdef SECOND_DIST 
                    C,
                    #endif 
                    #if SAVE_BC
                    nodeTypeSave,
                    hNodeType,
                    #endif
                    #if defined BC_FORCES && defined SAVE_BC_FORCES
                    h_BC_Fx,
                    h_BC_Fy,
                    h_BC_Fz,
                    #endif
                    step); 

                    printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);
                    if(console_flush){fflush(stdout);}
                    if(!ONLY_FINAL_MACRO){
                    saveMacr(rho,ux,uy,uz,
                    #ifdef NON_NEWTONIAN_FLUID
                    omega,
                    #endif
                    #ifdef SECOND_DIST 
                    C,
                    #endif 
                    #if SAVE_BC
                    nodeTypeSave,
                    #endif
                    #if defined BC_FORCES && defined SAVE_BC_FORCES
                    h_BC_Fx,
                    h_BC_Fy,
                    h_BC_Fz,
                    #endif
                    step);
                    }
                //}
            //}

            #ifdef BC_FORCES
                totalBcDrag(d_BC_Fx, d_BC_Fy, d_BC_Fz, step);
            #endif

        }

    } // end of the loop
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    checkCudaErrors(cudaDeviceSynchronize()); 
    checkCudaErrors(cudaMemcpy(h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    #endif

    linearMacr(h_fMom,rho,ux,uy,uz,
    #ifdef NON_NEWTONIAN_FLUID
    omega,
    #endif
    #ifdef SECOND_DIST 
    C,
    #endif 
    #if SAVE_BC
    nodeTypeSave,
    hNodeType,
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    h_BC_Fx,
    h_BC_Fy,
    h_BC_Fz,
    #endif
    step); 

    if(console_flush){fflush(stdout);}
    
    saveMacr(rho,ux,uy,uz,
    #ifdef NON_NEWTONIAN_FLUID
    omega,
    #endif
    #ifdef SECOND_DIST 
    C,
    #endif 
    #if SAVE_BC
    nodeTypeSave,
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    h_BC_Fx,
    h_BC_Fy,
    h_BC_Fz,
    #endif
    step);

    #ifdef PARTICLE_TRACER
        checkCudaErrors(cudaMemcpy(h_particlePos, d_particlePos, sizeof(dfloat3)*NUM_PARTICLES, cudaMemcpyDeviceToHost)); 
        saveParticleInfo(h_particlePos,step);
    #endif
    if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            
        checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_fGhostX_0,gGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_fGhostX_1,gGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_fGhostY_0,gGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_fGhostY_1,gGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_fGhostZ_0,gGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_fGhostZ_1,gGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        
        saveSimCheckpoint(fMom,gGhostX_0,gGhostX_1,gGhostY_0,gGhostY_1,gGhostZ_0,gGhostZ_1,
        #ifdef SECOND_DIST 
        g_fGhostX_0, g_fGhostX_1, g_fGhostY_0, g_fGhostY_1,g_fGhostZ_0, g_fGhostZ_1,
        #endif 
        &step);

    }
    checkCudaErrors(cudaDeviceSynchronize());
    #if MEAN_FLOW
            linearMacr(m_fMom,m_rho,m_ux,m_uy,m_uz,
            #ifdef NON_NEWTONIAN_FLUID
            omega,
            #endif
            #ifdef SECOND_DIST 
            m_c,
            #endif 
            #if SAVE_BC
            nodeTypeSave,
            hNodeType,
            #endif
            #if defined BC_FORCES && defined SAVE_BC_FORCES
            h_BC_Fx,
            h_BC_Fy,
            h_BC_Fz,
            #endif
            INT_MAX); 

            saveMacr(m_rho,m_ux,m_uy,m_uz,
            #ifdef NON_NEWTONIAN_FLUID
            omega,
            #endif
            #ifdef SECOND_DIST 
            m_c,
            #endif 
            #if SAVE_BC
            nodeTypeSave,
            #endif
            #if defined BC_FORCES && defined SAVE_BC_FORCES
            h_BC_Fx,
            h_BC_Fy,
            h_BC_Fz,
            #endif
            INT_MAX);
    #endif //MEAN_FLOW

    /* ------------------------------ POST ------------------------------ */
    //Calculate MLUPS
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaEventRecord(stop_step, 0));
    checkCudaErrors(cudaEventSynchronize(stop_step));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&(elapsedTime), start_step, stop_step));
    elapsedTime *= 0.001;
    size_t nodesUpdatedSync = (step) * NUMBER_LBM_NODES;
    dfloat MLUPS = (nodesUpdatedSync / 1e6) / elapsedTime;

    printf("MLUPS: %f\n",MLUPS);

    //save info file
    saveSimInfo(step,MLUPS);


    /* ------------------------------ FREE ------------------------------ */
    cudaFree(fMom);
    cudaFree(dNodeType);
    cudaFree(hNodeType);

    cudaFree(fGhostX_0);
    cudaFree(fGhostX_1);
    cudaFree(fGhostY_0);
    cudaFree(fGhostY_1);
    cudaFree(fGhostZ_0);
    cudaFree(fGhostZ_1);

    cudaFree(gGhostX_0);
    cudaFree(gGhostX_1);
    cudaFree(gGhostY_0);
    cudaFree(gGhostY_1);
    cudaFree(gGhostZ_0);
    cudaFree(gGhostZ_1);

    cudaFree(h_fMom);
    cudaFree(rho);
    cudaFree(ux);
    cudaFree(uy);
    cudaFree(uz);

    #ifdef SECOND_DIST 
    cudaFree(g_fGhostX_0);
    cudaFree(g_fGhostX_1);
    cudaFree(g_fGhostY_0);
    cudaFree(g_fGhostY_1);
    cudaFree(g_fGhostZ_0);
    cudaFree(g_fGhostZ_1);

    cudaFree(g_gGhostX_0);
    cudaFree(g_gGhostX_1);
    cudaFree(g_gGhostY_0);
    cudaFree(g_gGhostY_1);
    cudaFree(g_gGhostZ_0);
    cudaFree(g_gGhostZ_1);

    cudaFree(C);
    #endif 

    #if MEAN_FLOW
        cudaFree(m_fMom);
        cudaFree(m_rho);
        cudaFree(m_ux);
        cudaFree(m_uy);
        cudaFree(m_uz);
        #ifdef SECOND_DIST
        cudaFree(m_c);
        #endif
    #endif //MEAN_FLOW

    if(LOAD_CHECKPOINT){
        cudaFree(h_fGhostX_0);
        cudaFree(h_fGhostX_1);
        cudaFree(h_fGhostY_0);
        cudaFree(h_fGhostY_1);
        cudaFree(h_fGhostZ_0);
        cudaFree(h_fGhostZ_1);
        #ifdef SECOND_DIST 
        cudaFree(g_h_fGhostX_0);
        cudaFree(g_h_fGhostX_1);
        cudaFree(g_h_fGhostY_0);
        cudaFree(g_h_fGhostY_1);
        cudaFree(g_h_fGhostZ_0);
        cudaFree(g_h_fGhostZ_1);
        #endif 
    }

    #ifdef DENSITY_CORRECTION
        cudaFree(d_mean_rho);
        free(h_mean_rho);
    #endif
    #ifdef PARTICLE_TRACER
    cudaFree(h_particlePos);
    cudaFree(d_particlePos);
    #endif

    #if SAVE_BC
    cudaFree(nodeTypeSave);
    #endif


    return 0;



}
