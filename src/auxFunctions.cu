
#include "auxFunctions.cuh"

__host__
void mean_moment(dfloat *fMom, dfloat *meanMom, int m_index, size_t step){

    dfloat* sum;
    cudaMalloc((void**)&sum, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = NZ / nt_z;

    sumReductionThread << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sum,m_index);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z;

    int current_block_size = nb_x * nb_y * nb_z;
   
    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum, sum);
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
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum, sum);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sum, sizeof(dfloat), cudaMemcpyDeviceToHost)); 

    temp = (temp/(dfloat)NUMBER_LBM_NODES) - RHO_0; 
    //printf("step %d temp %e \n ",step, temp);
    checkCudaErrors(cudaMemcpy(meanMom, &temp, sizeof(dfloat), cudaMemcpyHostToDevice)); 
    cudaFree(sum);
    
}


__host__ 
void totalKineticEnergy(
    dfloat *fMom, 
    size_t step
){
    dfloat* sumKE;
    cudaMalloc((void**)&sumKE, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = NZ / nt_z;

    sumReductionThread_KE << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sumKE);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z;

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sumKE, sumKE);
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
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sumKE, sumKE);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sumKE, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    temp = (temp)/(U_MAX*U_MAX*NUMBER_LBM_NODES);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_totalKineticEnergy",strDataInfo.str(),step);
    cudaFree(sumKE);
}


__host__ 
void turbulentKineticEnergy(
    dfloat *fMom, 
    dfloat *m_fMom, 
    size_t step
){

    dfloat* sumTKE;
    cudaMalloc((void**)&sumTKE, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = NZ / nt_z;

    sumReductionThread_TKE << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom,m_fMom,sumTKE);

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sumTKE, sumTKE);
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
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sumTKE, sumTKE);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sumTKE, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    temp = (temp)/(U_MAX*U_MAX*NUMBER_LBM_NODES);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_turbulentKineticEnergy",strDataInfo.str(),step);
    cudaFree(sumTKE);

}
