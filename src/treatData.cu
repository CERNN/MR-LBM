#include "treatData.cuh"

__host__
void treatData(
    dfloat* h_fMom,
    dfloat* fMom,
    #if MEAN_FLOW
    dfloat* fMom_mean,
    #endif//MEAN_FLOW
    unsigned int step
){
    #if TREATFIELD
    computeNusseltNumber(h_fMom,fMom,step);
    #endif

    #if TREATPOINT
    probeExport(fMom, OMEGA_FIELD_PARAMS step);
    #endif
    #if TREATLINE
    velocityProfile(fMom,1,step);
    velocityProfile(fMom,2,step);
    velocityProfile(fMom,3,step);
    velocityProfile(fMom,4,step);
    velocityProfile(fMom,5,step);
    velocityProfile(fMom,6,step);
    #endif

    #ifdef TREAT_DATA_INCLUDE
    #include CASE_TREAT_DATA
    #endif

    //totalKineticEnergy(fMom,step);         
}

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
    temp = (temp)/(NUMBER_LBM_NODES);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_totalKineticEnergy",strDataInfo.str(),step);
    cudaFree(sumKE);
}

#ifdef CONVECTION_DIFFUSION_TRANSPORT
#ifdef CONFORMATION_TENSOR
__host__ 
void totalSpringEnergy(
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

    sumReductionThread_SE << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sumKE);

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
    temp = (temp/2.0) * nu_p * inv_lambda;
    temp = (temp)/(NUMBER_LBM_NODES);

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    strDataInfo <<"step,"<< step<< "," << temp;// << "," << mean_counter;



    saveTreatData("_totalSpringEnergy",strDataInfo.str(),step);
    cudaFree(sumKE);
}
#endif
#endif

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
    cudaFree(sum_BC_Fx);
    cudaFree(sum_BC_Fy);
    cudaFree(sum_BC_Fz);
   

};


__host__
void velocityProfile(
    dfloat* fMom,
    int dir_index,
    unsigned int step
){

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    strDataInfo <<"step "<< step;

    int x_loc,y_loc,z_loc;
    dfloat* ux;
    dfloat* uy;
    dfloat* uz;
    switch (dir_index)
    {
    case 1: //ux on y-direction
        
        checkCudaErrors(cudaMallocHost((void**)&(ux), sizeof(dfloat)));

        x_loc = NX/2;
        z_loc = NZ/2;
        
        for (y_loc = 0; y_loc < NY ; y_loc++){

            checkCudaErrors(cudaMemcpy(ux, fMom + idxMom(x_loc%BLOCK_NX,y_loc%BLOCK_NY, z_loc%BLOCK_NZ, 1, x_loc/BLOCK_NX, y_loc/BLOCK_NY, z_loc/BLOCK_NZ),
            sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo <<"\t"<< *ux;
        }
        saveTreatData("_ux_dy",strDataInfo.str(),step);

        cudaFree(ux);
        break;
    case 2: //uy on y-direction
        checkCudaErrors(cudaMallocHost((void**)&(uy), sizeof(dfloat)));

        x_loc = NX/2;
        z_loc = NZ/2;
        
        for (y_loc = 0; y_loc < NY ; y_loc++){

            checkCudaErrors(cudaMemcpy(uy, fMom + idxMom(x_loc%BLOCK_NX,y_loc%BLOCK_NY, z_loc%BLOCK_NZ, 2, x_loc/BLOCK_NX, y_loc/BLOCK_NY, z_loc/BLOCK_NZ),
            sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo <<"\t"<< *uy;
        }
        saveTreatData("_uy_dy",strDataInfo.str(),step);

        cudaFree(uy);
        break;
    case 3: //uz on y-direction
        checkCudaErrors(cudaMallocHost((void**)&(uz), sizeof(dfloat)));

        x_loc = NX/2;
        z_loc = NZ/2;
        
        for (y_loc = 0; y_loc < NY ; y_loc++){

            checkCudaErrors(cudaMemcpy(uz, fMom + idxMom(x_loc%BLOCK_NX,y_loc%BLOCK_NY, z_loc%BLOCK_NZ, 3, x_loc/BLOCK_NX, y_loc/BLOCK_NY, z_loc/BLOCK_NZ),
            sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo <<"\t"<< *uz;
        }
        saveTreatData("_uz_dy",strDataInfo.str(),step);

        cudaFree(uz);
        break;
    case 4: //ux on x-direction
        checkCudaErrors(cudaMallocHost((void**)&(ux), sizeof(dfloat)));

        y_loc = NY/2;
        z_loc = NZ/2;
        for (x_loc = 0; x_loc < NX ; x_loc++){

            checkCudaErrors(cudaMemcpy(ux, fMom + idxMom(x_loc%BLOCK_NX,y_loc%BLOCK_NY, z_loc%BLOCK_NZ, 1, x_loc/BLOCK_NX, y_loc/BLOCK_NY, z_loc/BLOCK_NZ),
            sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo <<"\t"<< *ux;
        }
        saveTreatData("_ux_dx",strDataInfo.str(),step);

        cudaFree(ux);
        break;
    case 5: //uy on x-direction
        checkCudaErrors(cudaMallocHost((void**)&(uy), sizeof(dfloat)));

        y_loc = NY/2;
        z_loc = NZ/2;
        for (x_loc = 0; x_loc < NX ; x_loc++){

            checkCudaErrors(cudaMemcpy(uy, fMom + idxMom(x_loc%BLOCK_NX,y_loc%BLOCK_NY, z_loc%BLOCK_NZ, 2, x_loc/BLOCK_NX, y_loc/BLOCK_NY, z_loc/BLOCK_NZ),
            sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo <<"\t"<< *uy;
        }
        saveTreatData("_uy_dx",strDataInfo.str(),step);

        cudaFree(uy);
        break;
    case 6: //uz on x-direction
        checkCudaErrors(cudaMallocHost((void**)&(uz), sizeof(dfloat)));

        y_loc = NY/2;
        z_loc = NZ/2;
        for (x_loc = 0; x_loc < NX ; x_loc++){

            checkCudaErrors(cudaMemcpy(uz, fMom + idxMom(x_loc%BLOCK_NX,y_loc%BLOCK_NY, z_loc%BLOCK_NZ, 3, x_loc/BLOCK_NX, y_loc/BLOCK_NY, z_loc/BLOCK_NZ),
            sizeof(dfloat), cudaMemcpyDeviceToHost));
            strDataInfo <<"\t"<< *ux;
        }
        saveTreatData("_uz_dx",strDataInfo.str(),step);

        cudaFree(uz);
        break;
    default:
        break;
    }
}


__host__
void probeExport(dfloat* fMom, OMEGA_FIELD_PARAMS_DECLARATION unsigned int step){
    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    strDataInfo <<"step "<< step;


    int probeNumber = 7;
    
    //probe locations
                //0        1       2       3        4       5   6
    int x[7] = {probe_x,(NX/4),(NX/4),(3*NX/4),(3*NX/4),(NX/4),(NX/4)};
    int y[7] = {probe_y,(NY/4),(3*NY/4),(3*NY/4),(NY/4),(NY/4),(NY/4)};
    int z[7] = {probe_z,probe_z,probe_z,probe_z,probe_z,(NZ_TOTAL/4),(3*NZ_TOTAL/4)};

    dfloat* rho;

    dfloat* ux;
    dfloat* uy;
    dfloat* uz;

    /*dfloat* mxx;
    dfloat* mxy;
    dfloat* mxz;
    dfloat* myy;
    dfloat* myz;
    dfloat* mzz;*/
    
    checkCudaErrors(cudaMallocHost((void**)&(rho), sizeof(dfloat)));    
    checkCudaErrors(cudaMallocHost((void**)&(ux), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(uy), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(uz), sizeof(dfloat)));    
    /*checkCudaErrors(cudaMallocHost((void**)&(mxx), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(mxy), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(mxz), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(myy), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(myz), sizeof(dfloat)));
    checkCudaErrors(cudaMallocHost((void**)&(mzz), sizeof(dfloat)));*/

    checkCudaErrors(cudaDeviceSynchronize());
    for(int i=0; i< probeNumber; i++){
        checkCudaErrors(cudaMemcpy(rho, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_RHO_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(ux , fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_UX_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(uy , fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_UY_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(uz , fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_UZ_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        /*checkCudaErrors(cudaMemcpy(mxx, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_MXX_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(mxy, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_MXY_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(mxz, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_MXZ_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(myy, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_MYY_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(myz, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_MYZ_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(mzz, fMom + idxMom(x[i]%BLOCK_NX, y[i]%BLOCK_NY, z[i]%BLOCK_NZ, M_MZZ_INDEX, x[i]/BLOCK_NX, y[i]/BLOCK_NY, z[i]/BLOCK_NZ),
        sizeof(dfloat), cudaMemcpyDeviceToHost));*/

        strDataInfo <<"\t"<< *ux << "\t" << *uy << "\t" << *uz;

    }
    saveTreatData("_probeData",strDataInfo.str(),step);




    cudaFree(rho);
    cudaFree(ux);
    cudaFree(uy);
    cudaFree(uz);
    /*cudaFree(mxx);
    cudaFree(mxy);
    cudaFree(mxz);
    cudaFree(myy);
    cudaFree(myz);
    cudaFree(mzz);*/

}



__host__
void computeNusseltNumber(
    dfloat* h_fMom,
    dfloat* fMom,
    unsigned int step
){
    //copy full macroscopic field
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    #ifdef THERMAL_MODEL

    
    int x0 = 0;
    int x1 = 1;
    int x2 = NX-1;
    int x3 = NX-2;
    dfloat C_x0;
    dfloat C_x1;
    dfloat C_x2;
    dfloat C_x3;
    dfloat Nu_sum = 0.0;


    for (int z = 0; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY-0;y++){
            C_x0 = h_fMom[idxMom(x0%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x0/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x1 = h_fMom[idxMom(x1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x2 = h_fMom[idxMom(x2%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x2/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x3 = h_fMom[idxMom(x3%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x3/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

            Nu_sum +=-(C_x1 - C_x0);
            Nu_sum +=(C_x3 - C_x2);
        }
    }

    Nu_sum /= (2*(NY-2)*NZ_TOTAL);
    Nu_sum = Nu_sum/(T_DELTA_T/L);

    strDataInfo <<"step,"<< step<< "," << Nu_sum;// << "," << mean_counter;
    saveTreatData("_Nu_mean",strDataInfo.str(),step);

    #endif 
}


__host__
void computeTurbulentEnergies(
    dfloat* h_fMom,
    dfloat* fMom,
    dfloat* fMom_mean,
    unsigned int step
){

    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);

    //Curent values
    dfloat t_ux0, t_uy0,t_uz0;
    dfloat t_mxx0,t_mxy0,t_mxz0,t_myy0,t_myz0,t_mzz0;


    dfloat Sxx = 0;
    dfloat Sxy = 0;
    dfloat Sxz = 0;
    dfloat Syy = 0;
    dfloat Syz = 0;
    dfloat Szz = 0;
    dfloat SS = 0;
    int count = 0;



    //fluctuation values
    dfloat f_ux = 0;
    dfloat f_uy = 0;
    dfloat f_uz = 0;

    dfloat f_Sxx = 0;
    dfloat f_Sxy = 0;
    dfloat f_Sxz = 0;
    dfloat f_Syy = 0;
    dfloat f_Syz = 0;
    dfloat f_Szz = 0;

    dfloat f_SS = 0;

    //mean values;
    dfloat m_ux = 0.0;
    dfloat m_uy = 0.0;
    dfloat m_uz = 0.0;

    dfloat m_Sxx = 0;
    dfloat m_Sxy = 0;
    dfloat m_Sxz = 0;
    dfloat m_Syy = 0;
    dfloat m_Syz = 0;
    dfloat m_Szz = 0;

#pragma warning(push)
#pragma warning(disable: 4804)
    dfloat mean_counter = 1.0/((dfloat)(step/MACR_SAVE)+1.0);
    count = 0;
#pragma warning(pop)

    //left side of the equation
    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                t_mxx0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mxy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mxz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_myy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_myz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mzz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                Sxx = (as2/(2*TAU))*(t_ux0*t_ux0-t_mxx0);
                Sxy = (as2/(2*TAU))*(t_ux0*t_uy0-t_mxy0);
                Sxz = (as2/(2*TAU))*(t_ux0*t_uz0-t_mxz0);

                Syy = (as2/(2*TAU))*(t_uy0*t_uy0-t_myy0);
                Syz = (as2/(2*TAU))*(t_uy0*t_uz0-t_myz0);

                Szz = (as2/(2*TAU))*(t_uz0*t_uz0-t_mzz0);
                SS += ( Sxx * Sxx + 
                        Syy * Syy + 
                        Szz * Szz + 2*(
                        Sxy * Sxy + 
                        Sxz * Sxz + 
                        Syz * Syz)) ;

                //STORE AND UPDATE MEANS

                //retrive mean values
                m_ux = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_uz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                
                m_Sxx = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Sxy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Sxz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Syy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Syz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                m_Szz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                //update and store mean values
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_ux + (t_ux0 - m_ux)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uy + (t_uy0 - m_uy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uz + (t_uz0 - m_uz)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Sxx + (Sxx - m_Sxx)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Sxy + (Sxy - m_Sxy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MXZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Sxz + (Sxz - m_Sxz)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Syy + (Syy - m_Syy)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MYZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Syz + (Syz - m_Syz)*(mean_counter);
                fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_MZZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_Szz + (Szz - m_Szz)*(mean_counter);
            
                f_ux = t_ux0 - m_ux;
                f_uy = t_uy0 - m_uy;
                f_uz = t_uz0 - m_uz;
                f_Sxx = Sxx - m_Sxx;
                f_Sxy = Sxy - m_Sxy;
                f_Sxz = Sxz - m_Sxz;
                f_Syy = Syy - m_Syy;
                f_Syz = Syz - m_Syz;
                f_Szz = Szz - m_Szz;

                f_SS += ( f_Sxx * f_Sxx + f_Syy * f_Syy + f_Szz * f_Szz + 2*( f_Sxy * f_Sxy + f_Sxz * f_Sxz + f_Syz * f_Syz));                        


                count++;
            }
        }
    }

    SS = SS/(N*N*N);
    f_SS = f_SS / (count);
    dfloat epsilon = 2*((TAU-0.5)/3)*f_SS;

    strDataInfo <<"step,"<< step<< "," << SS << "," << epsilon;
    saveTreatData("_turbulentData",strDataInfo.str(),step);
}