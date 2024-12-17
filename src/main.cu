#include "main.cuh"

using namespace std;

int main() {
    // Setup saving folder
    folderSetup();

    //set cuda device
    checkCudaErrors(cudaSetDevice(GPU_INDEX));

    //variable declaration
    dfloat* d_fMom;
    ghostInterfaceData ghostInterface;

    unsigned int* dNodeType;
    unsigned int* hNodeType;

    dfloat* h_fMom;
    dfloat* rho;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;
    
    
    #ifdef OMEGA_FIELD
    dfloat* omega;
    #endif

    #ifdef SECOND_DIST
    dfloat* C;
    #endif 

    #ifdef A_XX_DIST
    dfloat* Axx;
    #endif
    #ifdef A_XY_DIST
    dfloat* Axy;
    #endif
    #ifdef A_XZ_DIST
    dfloat* Axz;
    #endif
    #ifdef A_YY_DIST
    dfloat* Ayy;
    #endif
    #ifdef A_YZ_DIST
    dfloat* Ayz;
    #endif
    #ifdef A_ZZ_DIST
    dfloat* Azz;
    #endif

    #if NODE_TYPE_SAVE
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


    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    float** randomNumbers = nullptr; // useful for turbulence
    randomNumbers = (float**)malloc(sizeof(float*));

    allocateHostMemory(
        &h_fMom, &rho, &ux, &uy, &uz
        OMEGA_FIELD_PARAMS_PTR
        SECOND_DIST_PARAMS_PTR
        A_XX_DIST_PARAMS_PTR
        A_XY_DIST_PARAMS_PTR
        A_XZ_DIST_PARAMS_PTR
        A_YY_DIST_PARAMS_PTR
        A_YZ_DIST_PARAMS_PTR
        A_ZZ_DIST_PARAMS_PTR
        PARTICLE_TRACER_PARAMS_PTR(h_)
        MEAN_FLOW_PARAMS_PTR
        MEAN_FLOW_SECOND_DIST_PARAMS_PTR
        BC_FORCES_PARAMS_PTR(h_)
    );
    printf("Host Memory Allocated \n"); if(console_flush) fflush(stdout);
    /* -------------- ALLOCATION FOR GPU ------------- */
    allocateDeviceMemory(
        &d_fMom, &dNodeType, &ghostInterface
        PARTICLE_TRACER_PARAMS_PTR(d_)
        BC_FORCES_PARAMS_PTR(d_)
    );
    printf("Device Memory Allocated \n"); if(console_flush) fflush(stdout);
    #ifdef DENSITY_CORRECTION
        checkCudaErrors(cudaMallocHost((void**)&(h_mean_rho), sizeof(dfloat)));
        cudaMalloc((void**)&d_mean_rho, sizeof(dfloat));  
        printf("Density Correction Memory Allocated \n"); if(console_flush) fflush(stdout);
    #endif
    //printf("Allocated memory \n"); if(console_flush){fflush(stdout);}

    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef PARTICLE_TRACER
    cudaStream_t streamsPart[1];
    checkCudaErrors(cudaStreamCreate(&streamsPart[0]));
    #endif

    initializeDomain(ghostInterface,     
                     d_fMom, h_fMom, 
                     #if MEAN_FLOW
                     m_fMom,
                     #endif
                     hNodeType, dNodeType, randomNumbers, 
                     BC_FORCES_PARAMS(d_)
                     DENSITY_CORRECTION_PARAMS(h_)
                     DENSITY_CORRECTION_PARAMS(d_)
                     PARTICLE_TRACER_PARAMS_PTR(h_)
                     PARTICLE_TRACER_PARAMS_PTR(d_)
                     &step, gridBlock, threadBlock);

    printf("Domain Initialized\n"); if(console_flush) fflush(stdout);
    
    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    cudaEvent_t start, stop, start_step, stop_step;
    initializeCudaEvents(start, stop, start_step, stop_step);
    /* ------------------------------ LBM LOOP ------------------------------ */

    #ifdef DYNAMIC_SHARED_MEMORY
    cudaFuncSetAttribute(gpuMomCollisionStream, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEMORY_SIZE); // DOESNT WORK: DYNAMICALLY SHARED MEMORY HAS WORSE PERFORMANCE
    #endif

    /* --------------------------------------------------------------------- */
    /* ---------------------------- BEGIN LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */
    for (step=INI_STEP; step<N_STEPS;step++){

        int aux = step-INI_STEP;
        bool checkpoint = false;

        #ifdef DENSITY_CORRECTION
        mean_rho(d_fMom,step,d_mean_rho);
        #endif 

        bool save =false;
        bool reportSave = false;
        bool macrSave = false;

        if(aux != 0){
            if(REPORT_SAVE){ reportSave = !(step % REPORT_SAVE);}                
            if(MACR_SAVE){ macrSave   = !(step % MACR_SAVE);}
            if(MACR_SAVE || REPORT_SAVE){ save = (reportSave || macrSave);}
            if(CHECKPOINT_SAVE){ checkpoint = !(aux % CHECKPOINT_SAVE);}
        }
       
        gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(d_fMom, dNodeType,ghostInterface, DENSITY_CORRECTION_PARAMS(d_) BC_FORCES_PARAMS(d_) step, save); 

        #ifdef PARTICLE_TRACER
            checkCudaErrors(cudaDeviceSynchronize());
            updateParticlePos(d_particlePos, h_particlePos, d_fMom, streamsPart[0],step);
        #endif

        if(checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
            checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            interfaceCudaMemcpy(ghostInterface,ghostInterface.h_fGhost,ghostInterface.gGhost,cudaMemcpyDeviceToHost,QF);       
            #ifdef SECOND_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif    
            #ifdef A_XX_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Axx_h_fGhost,ghostInterface.Axx_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif       
            #ifdef A_XY_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Axy_h_fGhost,ghostInterface.Axy_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif           
            #ifdef A_XZ_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Axz_h_fGhost,ghostInterface.Axz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif           
            #ifdef A_YY_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayy_h_fGhost,ghostInterface.Ayy_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif           
            #ifdef A_YZ_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayz_h_fGhost,ghostInterface.Ayz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif           
            #ifdef A_ZZ_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Azz_h_fGhost,ghostInterface.Azz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif                 
            saveSimCheckpoint(d_fMom, ghostInterface, &step);
        }
       
        //swap interface pointers
        swapGhostInterfaces(ghostInterface);
        
        //save macroscopics

        //if (N_STEPS - step < 4*((int)turn_over_time)){
        if(reportSave){
            printf("\n--------------------------- Saving report %06d ---------------------------\n", step);
            treatData(h_fMom,d_fMom,
            #if MEAN_FLOW
            m_fMom,
            #endif //MEAN_FLOW
            step); 
        }
        if(macrSave){
            #if defined BC_FORCES && defined SAVE_BC_FORCES
            checkCudaErrors(cudaDeviceSynchronize()); 
            checkCudaErrors(cudaMemcpy(h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
            #endif
            //if (!(step%((int)turn_over_time/10))){
            //if((step>N_STEPS-80*(int)(MACR_SAVE))){ 
            //if((step%((int)(turn_over_time/2))) == 0){
                checkCudaErrors(cudaDeviceSynchronize()); 
                checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

                printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);
                if(console_flush){fflush(stdout);}
                //if(step > N_STEPS - 14000){
                if(!ONLY_FINAL_MACRO){
                    saveMacr(h_fMom,rho,ux,uy,uz, OMEGA_FIELD_PARAMS
                    #ifdef SECOND_DIST 
                    C,
                    #endif 
                    #ifdef A_XX_DIST 
                    Axx,
                    #endif 
                    #ifdef A_XY_DIST 
                    Axy,
                    #endif
                    #ifdef A_XZ_DIST 
                    Axz,
                    #endif
                    #ifdef A_YY_DIST 
                    Ayy,
                    #endif
                    #ifdef A_YZ_DIST 
                    Ayz,
                    #endif
                    #ifdef A_ZZ_DIST 
                    Azz,
                    #endif
                    NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) step);
                }
            //}

            #ifdef BC_FORCES
                totalBcDrag(d_BC_Fx, d_BC_Fy, d_BC_Fz, step);
            #endif
        }

    } 
    /* --------------------------------------------------------------------- */
    /* ------------------------------ END LOO ------------------------------ */
    /* --------------------------------------------------------------------- */

    checkCudaErrors(cudaDeviceSynchronize());

    //Calculate MLUPS

    dfloat MLUPS = recordElapsedTime(start_step, stop_step, step);
    printf("MLUPS: %f\n",MLUPS);
    
    /* ------------------------------ POST ------------------------------ */
    checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    checkCudaErrors(cudaDeviceSynchronize()); 
    checkCudaErrors(cudaMemcpy(h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    #endif



    if(console_flush){fflush(stdout);}
    
    saveMacr(h_fMom,rho,ux,uy,uz, OMEGA_FIELD_PARAMS 
    #ifdef SECOND_DIST 
    C,
    #endif 
    #ifdef A_XX_DIST 
    Axx,
    #endif 
    #ifdef A_XY_DIST 
    Axy,
    #endif
    #ifdef A_XZ_DIST 
    Axz,
    #endif
    #ifdef A_YY_DIST 
    Ayy,
    #endif
    #ifdef A_YZ_DIST 
    Ayz,
    #endif
    #ifdef A_ZZ_DIST 
    Azz,
    #endif
    NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(PREFIX) step);

    #ifdef PARTICLE_TRACER
        checkCudaErrors(cudaMemcpy(h_particlePos, d_particlePos, sizeof(dfloat3)*NUM_PARTICLES, cudaMemcpyDeviceToHost)); 
        saveParticleInfo(h_particlePos,step);
    #endif
    if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            
        checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        interfaceCudaMemcpy(ghostInterface,ghostInterface.h_fGhost,ghostInterface.gGhost,cudaMemcpyDeviceToHost,QF);    
        #ifdef SECOND_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif  
        #ifdef A_XX_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axx_h_fGhost,ghostInterface.Axx_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif 
        #ifdef A_XY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axy_h_fGhost,ghostInterface.Axy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif
        #ifdef A_XZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axz_h_fGhost,ghostInterface.Axz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif
        #ifdef A_YY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayy_h_fGhost,ghostInterface.Ayy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif
        #ifdef A_YZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayz_h_fGhost,ghostInterface.Ayz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif
        #ifdef A_ZZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Azz_h_fGhost,ghostInterface.Azz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif    
        saveSimCheckpoint(d_fMom,ghostInterface,&step);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    #if MEAN_FLOW
            saveMacr(m_fMom,m_rho,m_ux,m_uy,m_uz, OMEGA_FIELD_PARAMS
            #ifdef SECOND_DIST 
            m_c,
            #endif 
            NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) INT_MAX);
    #endif //MEAN_FLOW



    //save info file
    saveSimInfo(step,MLUPS);


    /* ------------------------------ FREE ------------------------------ */
    cudaFree(d_fMom);
    cudaFree(dNodeType);
    cudaFree(hNodeType);

    cudaFree(h_fMom);
    cudaFree(rho);
    cudaFree(ux);
    cudaFree(uy);
    cudaFree(uz);

    #ifdef SECOND_DIST 
    cudaFree(C);
    #endif 
    #ifdef A_XX_DIST 
    cudaFree(Axx);
    #endif 
    #ifdef A_XY_DIST 
    cudaFree(Axy);
    #endif
    #ifdef A_XZ_DIST 
    cudaFree(Axz);
    #endif
    #ifdef A_YY_DIST 
    cudaFree(Ayy);
    #endif
    #ifdef A_YZ_DIST 
    cudaFree(Ayz);
    #endif
    #ifdef A_ZZ_DIST 
    cudaFree(Azz);
    #endif

    interfaceFree(ghostInterface);

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



    #ifdef DENSITY_CORRECTION
        cudaFree(d_mean_rho);
        free(h_mean_rho);
    #endif
    #ifdef PARTICLE_TRACER
    cudaFree(h_particlePos);
    cudaFree(d_particlePos);
    #endif

    #if NODE_TYPE_SAVE
    cudaFree(nodeTypeSave);
    #endif
    return 0;
}