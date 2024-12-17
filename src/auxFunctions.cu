
#include "auxFunctions.cuh"



//TODO: there is some error in the sum when the blocks arent equal
__host__ 
void mean_rho(
    dfloat *fMom, 
    size_t step,
    dfloat *d_mean_rho
){
    dfloat* sumRho;
    cudaMalloc((void**)&sumRho, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = NZ / nt_z;

    sumReductionThread_rho << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sumRho);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z;

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sumRho, sumRho);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sumRho, sumRho);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sumRho, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    temp = (temp)/(NUMBER_LBM_NODES);
    checkCudaErrors(cudaMemcpy(d_mean_rho, &temp, sizeof(dfloat), cudaMemcpyHostToDevice)); 

    //std::ostringstream strDataInfo("");
    //strDataInfo << std::scientific;
    //strDataInfo << std::setprecision(6);
    //strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;
    //saveTreatData("_meanRho",strDataInfo.str(),step);

    cudaFree(sumRho);
}


__host__
void meanFlowComputation(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
){

    //current values
    dfloat t_ux0 = 0.0;
    dfloat t_uy0 = 0.0;
    dfloat t_uz0 = 0.0;

    #ifdef THERMAL_MODEL
    dfloat t_cc0 = 0.0;
    #endif

    //old mean values
    dfloat m_ux = 0.0;
    dfloat m_uy = 0.0;
    dfloat m_uz = 0.0;

    #ifdef THERMAL_MODEL
    dfloat m_cc = 0.0;
    #endif


    dfloat mean_counter = 1.0/((dfloat)(step/MACR_SAVE)+1.0);
    int count = 0;

    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){


                t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                #ifdef THERMAL_MODEL
                t_cc0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif

                //STORE AND UPDATE MEANS

                //retrive mean values
                m_ux = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                #ifdef THERMAL_MODEL
                m_cc = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif
                
                //update and store mean values
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_ux + (t_ux0 - m_ux)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uy + (t_uy0 - m_uy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uz + (t_uz0 - m_uz)*(mean_counter);

                #ifdef THERMAL_MODEL
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_cc + (t_cc0 - m_cc)*(mean_counter);
                #endif

                count++;

            }
        }
    }
}


