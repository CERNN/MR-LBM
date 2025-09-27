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

    #if NODE_TYPE_SAVE
    unsigned int* nodeTypeSave;
    #endif //NODE_TYPE_SAVE

    dfloat* h_fMom;
    dfloat* rho;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;
    
    
    #ifdef OMEGA_FIELD
    dfloat* omega;
    #endif //OMEGA_FIELD

    #ifdef SECOND_DIST
    dfloat* C;
    #endif //SECOND_DIST

    #ifdef A_XX_DIST
    dfloat* Axx;
    #endif //A_XX_DIST
    #ifdef A_XY_DIST
    dfloat* Axy;
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST
    dfloat* Axz;
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST
    dfloat* Ayy;
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST
    dfloat* Ayz;
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST
    dfloat* Azz;
    #endif //A_ZZ_DIST

    #ifdef DENSITY_CORRECTION
    dfloat* h_mean_rho;
    dfloat* d_mean_rho;
    #endif //DENSITY_CORRECTION

    #if MEAN_FLOW
        dfloat* m_fMom;
        dfloat* m_rho;
        dfloat* m_ux;
        dfloat* m_uy;
        dfloat* m_uz;
        #ifdef SECOND_DIST
        dfloat* m_c;
        #endif //SECOND_DIST
    #endif //MEAN_FLOW

    #ifdef BC_FORCES
        #ifdef SAVE_BC_FORCES
        dfloat* h_BC_Fx;
        dfloat* h_BC_Fy;
        dfloat* h_BC_Fz;
        #endif //SAVE_BC_FORCES

        dfloat* d_BC_Fx;
        dfloat* d_BC_Fy;
        dfloat* d_BC_Fz;
    #endif //_BC_FORCES


    /* ----------------- GRID AND THREADS DEFINITION FOR LBM ---------------- */
    dim3 threadBlock(BLOCK_NX, BLOCK_NY, BLOCK_NZ);
    dim3 gridBlock(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z);

    /* ------------------------- ALLOCATION FOR CPU ------------------------- */
    int step = 0;

    dfloat** randomNumbers = nullptr; // useful for turbulence
    randomNumbers = (dfloat**)malloc(sizeof(dfloat*));

   // Populations* pop;
   // Macroscopics* macr;

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
        MEAN_FLOW_PARAMS_PTR
        MEAN_FLOW_SECOND_DIST_PARAMS_PTR
        #if NODE_TYPE_SAVE
        , &nodeTypeSave
        #endif //NODE_TYPE_SAVE
        BC_FORCES_PARAMS_PTR(h_)
    );

    /* -------------- ALLOCATION FOR GPU ------------- */
    allocateDeviceMemory(
        &d_fMom, &dNodeType, &ghostInterface
        BC_FORCES_PARAMS_PTR(d_)
    );
    #ifdef DENSITY_CORRECTION
        checkCudaErrors(cudaMallocHost((void**)&(h_mean_rho), sizeof(dfloat)));
        cudaMalloc((void**)&d_mean_rho, sizeof(dfloat));  
    #endif //DENSITY_CORRECTION


    // Setup Streams
    cudaStream_t streamsLBM[1];
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    checkCudaErrors(cudaStreamCreate(&streamsLBM[0]));
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef PARTICLE_MODEL
    cudaStream_t streamsPart[1];
    checkCudaErrors(cudaStreamCreate(&streamsPart[0]));
    #endif //PARTICLE_MODEL

    step=INI_STEP;

    initializeDomain(ghostInterface,     
                     d_fMom, h_fMom, 
                     #if MEAN_FLOW
                     m_fMom,
                     #endif //MEAN_FLOW
                     hNodeType, dNodeType, randomNumbers, 
                     BC_FORCES_PARAMS(d_)
                     DENSITY_CORRECTION_PARAMS(h_)
                     DENSITY_CORRECTION_PARAMS(d_)
                     &step, gridBlock, threadBlock);

    int ini_step = step;

    printf("Domain Initialized. Starting simulation\n"); if(console_flush) fflush(stdout);
    
    #ifdef PARTICLE_MODEL
        //memory allocation for particles in host and device
        ParticlesSoA particlesSoA;
        Particle *particles;
        particles = (Particle*) malloc(sizeof(Particle)*NUM_PARTICLES);
        
        // particle initialization with position, velocity, and solver method
        initializeParticle(particlesSoA, particles, &step, gridBlock, threadBlock);

        saveParticlesInfo(&particlesSoA, step);

    #endif //PARTICLE_MODEL

    /* ------------------------------ TIMER EVENTS  ------------------------------ */
    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    cudaEvent_t start, stop, start_step, stop_step;
    initializeCudaEvents(start, stop, start_step, stop_step);
    /* ------------------------------ LBM LOOP ------------------------------ */

    #ifdef DYNAMIC_SHARED_MEMORY
        int maxShared;
        cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (MAX_SHARED_MEMORY_SIZE > maxShared) {
            printf("Requested %d bytes exceeds device max %d bytes\n", MAX_SHARED_MEMORY_SIZE, maxShared);
        }else{
            printf("Using %d bytes of dynamic shared memory of a max of %d bytes\n", MAX_SHARED_MEMORY_SIZE, maxShared);
            cudaFuncSetAttribute(&gpuMomCollisionStream, cudaFuncAttributeMaxDynamicSharedMemorySize DYNAMIC_SHARED_MEMORY_PARAMS); // DOESNT WORK: DYNAMICALLY SHARED MEMORY HAS WORSE PERFORMANCE
        }
    #endif //DYNAMIC_SHARED_MEMORY

    /* --------------------------------------------------------------------- */
    /* ---------------------------- BEGIN LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */
    for (;step<N_STEPS;step++){ // step is already initialized

        int aux = step-INI_STEP;
        bool checkpoint = false;

        #ifdef DENSITY_CORRECTION
        mean_rho(d_fMom,step,d_mean_rho);
        #endif //DENSITY_CORRECTION

        bool save =false;
        bool reportSave = false;
        bool macrSave = false;
        bool particleSave = false;

#pragma warning(push)
#pragma warning(disable: 4804)
        if(aux != 0){
            if(REPORT_SAVE){ reportSave = !(step % REPORT_SAVE);}                
            if(MACR_SAVE){ macrSave   = !(step % MACR_SAVE);}
            if(MACR_SAVE || REPORT_SAVE){ save = (reportSave || macrSave);}
            if(CHECKPOINT_SAVE){ checkpoint = !(aux % CHECKPOINT_SAVE);}
            #ifdef PARTICLE_MODEL
                if(PARTICLES_SAVE){ particleSave = !(aux % PARTICLES_SAVE);}
            #endif //PARTICLE MODEL
        }
#pragma warning(pop)


        gpuMomCollisionStream << <gridBlock, threadBlock DYNAMIC_SHARED_MEMORY_PARAMS>> >(d_fMom, dNodeType,ghostInterface, DENSITY_CORRECTION_PARAMS(d_) BC_FORCES_PARAMS(d_) step, save);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        //swap interface pointers
        swapGhostInterfaces(ghostInterface);
        
        #ifdef LOCAL_FORCES
            gpuResetMacroForces<<<gridBlock, threadBlock>>>(d_fMom);
        #endif //LOCAL_FORCES

        #ifdef PARTICLE_MODEL
            particleSimulation(&particlesSoA,d_fMom,streamsPart,step);
        #endif //PARTICLE_MODEL


        if(checkpoint){
            printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            // throwing a warning for being used without being initialized. But does not matter since we are overwriting it;
            checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
            interfaceCudaMemcpy(ghostInterface,ghostInterface.h_fGhost,ghostInterface.fGhost,cudaMemcpyDeviceToHost,QF);       
            #ifdef SECOND_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //SECOND_DIST
            #ifdef A_XX_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Axx_h_fGhost,ghostInterface.Axx_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_XX_DIST     
            #ifdef A_XY_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Axy_h_fGhost,ghostInterface.Axy_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_XX_DIST        
            #ifdef A_XZ_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Axz_h_fGhost,ghostInterface.Axz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_XZ_DIST
            #ifdef A_YY_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayy_h_fGhost,ghostInterface.Ayy_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_YY_DIST        
            #ifdef A_YZ_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayz_h_fGhost,ghostInterface.Ayz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_YZ_DIST      
            #ifdef A_ZZ_DIST 
            interfaceCudaMemcpy(ghostInterface,ghostInterface.Azz_h_fGhost,ghostInterface.Azz_fGhost,cudaMemcpyDeviceToHost,GF);
            #endif //A_ZZ_DIST              
            saveSimCheckpoint(h_fMom, ghostInterface, &step);

            #ifdef PARTICLE_MODEL
                printf("Starting saveSimCheckpointParticle...\t"); fflush(stdout);
                saveSimCheckpointParticle(particlesSoA, &step);
            #endif //PARTICLE_MODEL
            

        }
       
        
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
            #endif //BC_FORCES && SAVE_BC_FORCES
            //if (!(step%((int)turn_over_time/10))){
            //if((step>N_STEPS-80*(int)(MACR_SAVE))){ 
            //if((step%((int)(turn_over_time/2))) == 0){
                checkCudaErrors(cudaDeviceSynchronize()); 
                checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

                printf("\n--------------------------- Saving macro %06d ---------------------------\n", step);
                if(console_flush){fflush(stdout);}
                //if(step > N_STEPS - 14000){
                if(!ONLY_FINAL_MACRO){
                    saveMacr(h_fMom,rho,ux,uy,uz, hNodeType, OMEGA_FIELD_PARAMS
                    #ifdef SECOND_DIST 
                    C,
                    #endif //SECOND_DIST
                    #ifdef A_XX_DIST 
                    Axx,
                    #endif //A_XX_DIST
                    #ifdef A_XY_DIST 
                    Axy,
                    #endif //A_XY_DIST
                    #ifdef A_XY_DIST 
                    Axz,
                    #endif //A_XY_DIST
                    #ifdef A_YY_DIST 
                    Ayy,
                    #endif //A_YY_DIST
                    #ifdef A_YZ_DIST 
                    Ayz,
                    #endif //A_YY_DIST
                    #ifdef A_ZZ_DIST 
                    Azz,
                    #endif //A_ZZ_DIST
                    NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) step);
                }
            //}

            #ifdef BC_FORCES
                totalBcDrag(d_BC_Fx, d_BC_Fy, d_BC_Fz, step);
            #endif //BC_FORCES
        }

        #ifdef PARTICLE_MODEL
            if (particleSave){
                printf("\n------------------------- Saving particles %06d -------------------------\n", step);
                if(console_flush){fflush(stdout);}
                saveParticlesInfo(&particlesSoA, step);
            }
        #endif //PARTICLE_MODEL

    } 
    /* --------------------------------------------------------------------- */
    /* ------------------------------ END LOOP ----------------------------- */
    /* --------------------------------------------------------------------- */

    checkCudaErrors(cudaDeviceSynchronize());

    //Calculate MLUPS

    dfloat MLUPS = recordElapsedTime(start_step, stop_step, step, ini_step);
    printf("MLUPS: %f\n",MLUPS);
    
    /* ------------------------------ POST ------------------------------ */
    checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));

    #if defined BC_FORCES && defined SAVE_BC_FORCES
    checkCudaErrors(cudaDeviceSynchronize()); 
    checkCudaErrors(cudaMemcpy(h_BC_Fx, d_BC_Fx, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_BC_Fy, d_BC_Fy, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_BC_Fz, d_BC_Fz, MEM_SIZE_SCALAR, cudaMemcpyDeviceToHost));
    #endif //BC_FORCES && SAVE_BC_FORCES



    if(console_flush){fflush(stdout);}
    
    saveMacr(h_fMom,rho,ux,uy,uz, hNodeType, OMEGA_FIELD_PARAMS 
    #ifdef SECOND_DIST 
    C,
    #endif //SECOND_DIST
    #ifdef A_XX_DIST 
    Axx,
    #endif //A_XX_DIST
    #ifdef A_XY_DIST 
    Axy,
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST 
    Axz,
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST 
    Ayy,
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST 
    Ayz,
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST 
    Azz,
    #endif //A_ZZ_DIST
    NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) step);


    /*if(CHECKPOINT_SAVE){
        printf("\n--------------------------- Saving checkpoint %06d ---------------------------\n", step);fflush(stdout);
            
        checkCudaErrors(cudaMemcpy(h_fMom, d_fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
        interfaceCudaMemcpy(ghostInterface,ghostInterface.h_fGhost,ghostInterface.gGhost,cudaMemcpyDeviceToHost,QF);    
        #ifdef SECOND_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.g_h_fGhost,ghostInterface.g_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //SECOND_DIST 
        #ifdef A_XX_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axx_h_fGhost,ghostInterface.Axx_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axy_h_fGhost,ghostInterface.Axy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Axz_h_fGhost,ghostInterface.Axz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayy_h_fGhost,ghostInterface.Ayy_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Ayz_h_fGhost,ghostInterface.Ayz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        interfaceCudaMemcpy(ghostInterface,ghostInterface.Azz_h_fGhost,ghostInterface.Azz_fGhost,cudaMemcpyDeviceToHost,GF);
        #endif //A_ZZ_DIST   
        saveSimCheckpoint(d_fMom,ghostInterface,&step);
    }*/
    checkCudaErrors(cudaDeviceSynchronize());
    #if MEAN_FLOW
            saveMacr(m_fMom,m_rho,m_ux,m_uy,m_uz, hNodeType, OMEGA_FIELD_PARAMS
            #ifdef SECOND_DIST 
            m_c,
            #endif  //SECOND_DIST
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

    // Free particle
    #ifdef PARTICLE_MODEL
    free(particles);
    particlesSoA.freeNodesAndCenters();
    #endif //PARTICLE_MODEL
    
    #ifdef SECOND_DIST 
    cudaFree(C);
    #endif //SECOND_DIST
    #ifdef A_XX_DIST 
    cudaFree(Axx);
    #endif //A_XX_DIST
    #ifdef A_XY_DIST 
    cudaFree(Axy);
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST 
    cudaFree(Axz);
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST 
    cudaFree(Ayy);
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST 
    cudaFree(Ayz);
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST 
    cudaFree(Azz);
    #endif //A_ZZ_DIST

    interfaceFree(ghostInterface);

    #if MEAN_FLOW
        cudaFree(m_fMom);
        cudaFree(m_rho);
        cudaFree(m_ux);
        cudaFree(m_uy);
        cudaFree(m_uz);
        #ifdef SECOND_DIST
        cudaFree(m_c);
        #endif //MEAN_FLOW
    #endif //MEAN_FLOW



    #ifdef DENSITY_CORRECTION
        cudaFree(d_mean_rho);
        free(h_mean_rho);
    #endif //DENSITY_CORRECTION


    #if NODE_TYPE_SAVE
    cudaFree(nodeTypeSave);
    #endif //NODE_TYPE_SAVE
    return 0;
}