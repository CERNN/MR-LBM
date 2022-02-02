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

__host__ __device__
void interfaceSwap(dfloat* &pt1, dfloat* &pt2){
  dfloat *temp = pt1;
  pt1 = pt2;
  pt2 = temp;
} 

int main() {

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

    char* dNodeType;


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
    cudaMalloc((void**)&dNodeType, sizeof(char) * NUMBER_LBM_NODES);  

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
    printf("Allocated memory \n");fflush(stdout);



    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());

    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- LBM INITIALIZATION ------------------------- */
    printf("Moments initialized \n");fflush(stdout);
    gpuInitialization_nodeType << <gridBlock, threadBlock >> >(dNodeType);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuInitialization_pop << <gridBlock, threadBlock >> >(fMom,fGhostX_0,fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1);
    printf("Interface Populations initialized \n");fflush(stdout);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(gGhostX_0, fGhostX_0, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostX_1, fGhostX_1, sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostY_0, fGhostY_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostY_1, fGhostY_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostZ_0, fGhostZ_0, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(gGhostZ_1, fGhostZ_1, sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    size_t step = 0;
    printf("%d,",step); fflush(stdout);


    bool save = false;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    linearMacr(h_fMom,rho,ux,uy,uz,step);

    printf("Initializing loop \n");fflush(stdout);
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    cudaEvent_t start, stop, start_step, stop_step;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventCreate(&start_step));
    checkCudaErrors(cudaEventCreate(&stop_step));

    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaEventRecord(start_step, 0));
    /* ------------------------------ LBM LOOP ------------------------------ */
    

    for (step=1; step<N_STEPS;step++){
        save =false;

        if(MACR_SAVE)
            save = !(step % MACR_SAVE);

        gpuMomCollisionStream << <gridBlock, threadBlock >> > (fMom,dNodeType,
        fGhostX_0,fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1,
        gGhostX_0,gGhostX_1,gGhostY_0,gGhostY_1,gGhostZ_0,gGhostZ_1);


        
        //swap interface pointers
        checkCudaErrors(cudaDeviceSynchronize());
        interfaceSwap(fGhostX_0,gGhostX_0);
        interfaceSwap(fGhostX_1,gGhostX_1);
        interfaceSwap(fGhostY_0,gGhostY_0);
        interfaceSwap(fGhostY_1,gGhostY_1);
        interfaceSwap(fGhostZ_0,gGhostZ_0);
        interfaceSwap(fGhostZ_1,gGhostZ_1);
        

        //save macroscopics

        if(save){
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaDeviceSynchronize());
            
            printf("step %d \n",step);
            linearMacr(h_fMom,rho,ux,uy,uz,step); 
            fflush(stdout);
            printf("------------------------------------------------------------------------\n");
            saveMacr(rho,ux,uy,uz,step);
        }

    }
    checkCudaErrors(cudaDeviceSynchronize());
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
    saveSimInfo(step);


    /* ------------------------------ FREE ------------------------------ */
    cudaFree(fMom);
    cudaFree(dNodeType);
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

    return 0;



}
