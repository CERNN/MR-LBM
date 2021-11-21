#include <stdio.h>
#include <stdlib.h>

// CUDA INCLUDE
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// FILES INCLUDES
#include "var.h"
#include "errorDef.h"
//#include "structs.h"
//#include "globalFunctions.h"
#include "lbmInitialization.cuh"
#include "mlbm.cuh"
#include "saveData.cuh"

using namespace std;

int main() {

    dfloat* fMom;
    dfloat* fGhostX_0;
    dfloat* fGhostX_1;
    dfloat* fGhostY_0; 
    dfloat* fGhostY_1;
    dfloat* fGhostZ_0; 
    dfloat* fGhostZ_1;



    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    dfloat* h_fMom;
    dfloat* rho;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;

    checkCudaErrors(cudaMallocHost((void**)&(h_fMom), MEM_SIZE_MOM));
    checkCudaErrors(cudaMallocHost((void**)&(rho), MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)&(ux), MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)&(uy), MEM_SIZE_SCALAR));
    checkCudaErrors(cudaMallocHost((void**)&(uz), MEM_SIZE_SCALAR));


    // Setup saving folder
    folderSetup();

    /* -------------- ALLOCATION AND CONFIGURATION FOR EACH GPU ------------- */

    cudaMalloc((void**)&fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS);    
    cudaMalloc((void**)&fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);    
    cudaMalloc((void**)&fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF);    
    cudaMalloc((void**)&fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);    
    cudaMalloc((void**)&fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF);    
    cudaMalloc((void**)&fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);    
    cudaMalloc((void**)&fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF);    


    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- LBM INITIALIZATION ------------------------- */
    gpuInitialization_mom << <gridBlock, threadBlock >> >(fMom);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuInitialization_pop << <gridBlock, threadBlock >> >(fMom,fGhostX_0,fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaSetDevice(0));
    cudaEvent_t start, stop, start_step, stop_step;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));
    /* ------------------------------ LBM LOOP ------------------------------ */
    
    size_t step = 0;
    bool save = false;
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            linearMacr(h_fMom,rho,ux,uy,uz);
            //saveMacr(rho,ux,uy,uz,step);
    for (step=1; step<N_STEPS;step++){
        save =false;

        if(MACR_SAVE)
            save = !(step % MACR_SAVE);

        gpuMomCollisionStream << <gridBlock, threadBlock >> > (fMom,fGhostX_0,fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1);
        fflush(stdout);

        //save macroscopics
        if(save){
            //printf("step %d \n",step);
            //printf("------------------------------------------------------------------------\n");
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            linearMacr(h_fMom,rho,ux,uy,uz);
            saveMacr(rho,ux,uy,uz,step);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    /* ------------------------------ POST ------------------------------ */
    //Calculate MLUPS
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaEventRecord(stop_step, 0));
    checkCudaErrors(cudaEventSynchronize(stop_step));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&(elapsedTime), start_step, stop_step));
    elapsedTime *= 0.001;
    size_t nodesUpdatedSync = (step) * NUMBER_LBM_NODES;
    dfloat MLUPS = (nodesUpdatedSync / 1e6) / elapsedTime;

    printf("MLUPS: %f\n",MLUPS);



    /* ------------------------------ FREE ------------------------------ */
    cudaFree(fMom);
    cudaFree(fGhostX_0);
    cudaFree(fGhostX_1);
    cudaFree(fGhostY_0);
    cudaFree(fGhostY_1);
    cudaFree(fGhostZ_0);
    cudaFree(fGhostZ_1);

    cudaFree(h_fMom);
    cudaFree(rho);
    cudaFree(ux);
    cudaFree(uy);
    cudaFree(uz);

    return 0;



}
