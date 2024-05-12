
#include "auxFunctions.cuh"

__host__
void mean_moment(dfloat *fMom, dfloat *meanMom, int m_index, size_t step, int target){

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

    if (m_index == M_RHO_INDEX){
        temp = (temp/(dfloat)NUMBER_LBM_NODES); 
    }else{
        temp = (temp/(dfloat)NUMBER_LBM_NODES);
    }
                
    if (target == 0){
        checkCudaErrors(cudaMemcpy(meanMom, &temp, sizeof(dfloat), cudaMemcpyHostToDevice)); 
    }
    else{
        checkCudaErrors(cudaMemcpy(meanMom, &temp, sizeof(dfloat), cudaMemcpyHostToHost)); 
    }


    cudaFree(sum);


    
}

__host__
void mean_rho(
    dfloat *fMom, 
    size_t step
){

    dfloat* sum;
    cudaMalloc((void**)&sum, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = NZ / nt_z;

    sumReductionThread << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sum,M_RHO_INDEX);

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


    temp = (temp/(dfloat)NUMBER_LBM_NODES); 

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_meanRho",strDataInfo.str(),step);

    cudaFree(sum);
}

//TODO: there is some error in the sum when the blocks arent equal
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


void totalBcDrag(
    dfloat *d_BC_Fx, 
    dfloat* d_BC_Fy, 
    dfloat* d_BC_Fz, 
    size_t step
){
    dfloat* sum_BC_Fx;
    dfloat* sum_BC_Fy;
    dfloat* sum_BC_Fz;

    dfloat* h_BC_Fx;
    dfloat* h_BC_Fy;
    dfloat* h_BC_Fz;
   
    cudaMalloc((void**)&sum_BC_Fx, NUM_BLOCK * sizeof(dfloat));
    cudaMalloc((void**)&sum_BC_Fy, NUM_BLOCK * sizeof(dfloat));
    cudaMalloc((void**)&sum_BC_Fz, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NY / nt_y;
    int nb_z = NZ / nt_z;

    sumReductionScalar << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (d_BC_Fx, sum_BC_Fx);
    sumReductionScalar << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (d_BC_Fy, sum_BC_Fy);
    sumReductionScalar << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (d_BC_Fz, sum_BC_Fz);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z;

    int current_block_size = nb_x * nb_y * nb_z;

    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum_BC_Fx, sum_BC_Fx);
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum_BC_Fy, sum_BC_Fy);
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum_BC_Fz, sum_BC_Fz);
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
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum_BC_Fx, sum_BC_Fx);
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum_BC_Fy, sum_BC_Fy);
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum_BC_Fz, sum_BC_Fz);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp_x, temp_y, temp_z;
    
    checkCudaErrors(cudaMemcpy(&temp_x, sum_BC_Fx, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&temp_y, sum_BC_Fy, sizeof(dfloat), cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&temp_z, sum_BC_Fz, sizeof(dfloat), cudaMemcpyDeviceToHost)); 


    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp_x<< "," << temp_y<< "," << temp_z;// << "," << mean_counter;

    saveTreatData("_totalBcDrag",strDataInfo.str(),step);
    cudaFree(h_BC_Fx);
    cudaFree(h_BC_Fy);
    cudaFree(h_BC_Fz);
   

};