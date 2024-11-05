
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

