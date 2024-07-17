#include "saveData.cuh"
#ifdef NON_NEWTONIAN_FLUID
#include "nnf.h"
#endif

__host__
void treatData(
    dfloat* h_fMom,
    dfloat* fMom,
    #if MEAN_FLOW
    dfloat* fMom_mean,
    #endif
    unsigned int step
){

    //copy full macroscopic field
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());


    #ifdef THERMAL_MODEL
    std::ostringstream strDataInfo("");
    strDataInfo << std::scientific;
    strDataInfo << std::setprecision(6);
    
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
            C_x0 = h_fMom[idxMom(x0%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_C_INDEX, x0/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x1 = h_fMom[idxMom(x1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_C_INDEX, x1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x2 = h_fMom[idxMom(x2%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_C_INDEX, x2/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            C_x3 = h_fMom[idxMom(x3%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_C_INDEX, x3/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

            Nu_sum +=-(C_x1 - C_x0);
            Nu_sum +=(C_x3 - C_x2);
        }
    }

    Nu_sum /= (2*(NY-2)*NZ_TOTAL);
    Nu_sum = Nu_sum/(T_DELTA_T/L);

    strDataInfo <<"step,"<< step<< "," << Nu_sum;// << "," << mean_counter;
    saveTreatData("_Nu_mean",strDataInfo.str(),step);
    #endif

    /*
    dfloat t_ux0, t_ux1;
    dfloat m_ux0_s, m_ux1_s;
    int y0 = NY-1;
    int y1 = NY-2;
    int count = 0;
    m_ux0_s = 0.0;
    m_ux1_s = 0.0;

    //right side of the equation 10
    for (int z = 0 ; z <NZ_TOTAL-1 ; z++){
        for (int x = 0; x< NX-1;x++){
            t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y0%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y0/BLOCK_NY, z/BLOCK_NZ)];
            t_ux1 = h_fMom[idxMom(x%BLOCK_NX, y1%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y1/BLOCK_NY, z/BLOCK_NZ)];

            m_ux0_s += (t_ux0 * t_ux0);
            m_ux1_s += (t_ux1 * t_ux1);
            count++;
        }
    }
    m_ux0_s /= count;
    m_ux1_s /= count;


    dfloat LS = (m_ux0_s-m_ux1_s);
    LS = LS/(4*N);
    */

    //LEFT SIDE
/*
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


    #if MEAN_FLOW
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

    dfloat m_ux = 0.0;
    dfloat m_uy = 0.0;
    dfloat m_uz = 0.0;

    dfloat m_Sxx = 0;
    dfloat m_Sxy = 0;
    dfloat m_Sxz = 0;
    dfloat m_Syy = 0;
    dfloat m_Syz = 0;
    dfloat m_Szz = 0;
    #endif 

    dfloat mean_counter = 1.0/((dfloat)(step/MACR_SAVE)+1.0);
    count = 0;
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

                #if MEAN_FLOW
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


                #endif
                count++;

            }
        }
    }

    SS = SS/(N*N*N);
    #if MEAN_FLOW
    f_SS = f_SS / (count);
    #endif


    int y0 = NY-1;
    int y1 = NY-2;
    dfloat t_ux1;
    
    dfloat mean_prod = 0.0;
    for (int z = 0 ; z <NZ_TOTAL ; z++){
        for (int x = 0; x< NX;x++){
            t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y0%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y0/BLOCK_NY, z/BLOCK_NZ)];
            t_ux1 = h_fMom[idxMom(x%BLOCK_NX, y1%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y1/BLOCK_NY, z/BLOCK_NZ)];
            mean_prod +=(t_ux0*t_ux0-t_ux1*t_ux1)/4;
        }
    }
    mean_prod = mean_prod/(N*N*N);

    #if MEAN_FLOW
    dfloat epsilon = 2*((TAU-0.5)/3)*f_SS;
    #endif




    //printf("%0.7e\t%0.7e\t%0.7e\n",LS,SS,SS/LS);
    // step << total_energy_dissipated, total_energy_produced, error , epsilon, omega
    strDataInfo <<"step,"<< step<< "," << SS << "," << mean_prod << "," << abs(SS/mean_prod - 1.0);// << "," << mean_counter;
    #if MEAN_FLOW
        strDataInfo <<"," <<  epsilon;
    #endif



    saveTreatData("_treatData",strDataInfo.str(),step);
    */
}

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
void probeExport(
    dfloat* fMom,
    #ifdef NON_NEWTONIAN_FLUID
    dfloat* omega,
    #endif
    unsigned int step
){
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
void linearMacr(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    #ifdef NON_NEWTONIAN_FLUID
    dfloat* omega,
    #endif
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif
    #if SAVE_BC
    dfloat* nodeTypeSave,
    unsigned int* hNodeType,
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    dfloat* h_BC_Fx,
    dfloat* h_BC_Fy,
    dfloat* h_BC_Fz,
    #endif
    unsigned int step
){
    size_t indexMacr;
    for(int z = 0; z< NZ;z++){
        ///printf("z %d \n", z);
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                indexMacr = idxScalarGlobal(x,y,z);

                rho[indexMacr] = RHO_0+h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                ux[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                #ifdef NON_NEWTONIAN_FLUID
                omega[indexMacr] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]; 
                #endif

                #ifdef SECOND_DIST 
                C[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif
                
                #if SAVE_BC
                nodeTypeSave[indexMacr] = (dfloat)hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]; 
                #endif
                //data += rho[indexMacr]*(ux[indexMacr]*ux[indexMacr] + uy[indexMacr]*uy[indexMacr] + uz[indexMacr]*uz[indexMacr]);
                //meanRho += rho[indexMacr];
            }
        }
    }


    #if defined BC_FORCES && defined SAVE_BC_FORCES
        dfloat* temp_x; 
        dfloat* temp_y;
        dfloat* temp_z;
        checkCudaErrors(cudaMallocHost((void**)&(temp_x), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(temp_y), MEM_SIZE_SCALAR));
        checkCudaErrors(cudaMallocHost((void**)&(temp_z), MEM_SIZE_SCALAR));


        for(int z = 0; z< NZ;z++){
            for(int y = 0; y< NY;y++){
                for(int x = 0; x< NX;x++){
                    indexMacr = idxScalarGlobal(x,y,z);
                    temp_x[indexMacr] = h_BC_Fx[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                    temp_y[indexMacr] = h_BC_Fy[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                    temp_z[indexMacr] = h_BC_Fz[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                }
            }
        }

        checkCudaErrors(cudaMemcpy(h_BC_Fx, temp_x, MEM_SIZE_SCALAR, cudaMemcpyHostToHost));
        checkCudaErrors(cudaMemcpy(h_BC_Fy, temp_y, MEM_SIZE_SCALAR, cudaMemcpyHostToHost));
        checkCudaErrors(cudaMemcpy(h_BC_Fz, temp_z, MEM_SIZE_SCALAR, cudaMemcpyHostToHost));


        cudaFreeHost(temp_x);
        cudaFreeHost(temp_y);
        cudaFreeHost(temp_z);
    #endif

}


__host__
void saveMacr(
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    #ifdef NON_NEWTONIAN_FLUID
    dfloat* omega,
    #endif
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif
    #if SAVE_BC
    dfloat* nodeTypeSave,
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    dfloat* h_BC_Fx,
    dfloat* h_BC_Fy,
    dfloat* h_BC_Fz,
    #endif
    unsigned int nSteps
){
// Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz;
    std::string strFileOmega;
    std::string strFileC;
    std::string strFileBc; 
    std::string strFileFx, strFileFy, strFileFz;

    strFileRho = getVarFilename("rho", nSteps, ".bin");
    strFileUx = getVarFilename("ux", nSteps, ".bin");
    strFileUy = getVarFilename("uy", nSteps, ".bin");
    strFileUz = getVarFilename("uz", nSteps, ".bin");
    #ifdef NON_NEWTONIAN_FLUID
    strFileOmega = getVarFilename("omega", nSteps, ".bin");
    #endif
    #ifdef SECOND_DIST 
    strFileC = getVarFilename("C", nSteps, ".bin");
    #endif
    #if SAVE_BC
    strFileBc = getVarFilename("bc", nSteps, ".bin");
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    strFileFx = getVarFilename("fx", nSteps, ".bin");
    strFileFy = getVarFilename("fy", nSteps, ".bin");
    strFileFz = getVarFilename("fz", nSteps, ".bin");
    #endif
    // saving files
    saveVarBin(strFileRho, rho, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUx, ux, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUy, uy, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUz, uz, MEM_SIZE_SCALAR, false);
    #ifdef NON_NEWTONIAN_FLUID
    saveVarBin(strFileOmega, omega, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef SECOND_DIST
    saveVarBin(strFileC, C, MEM_SIZE_SCALAR, false);
    #endif
    #if SAVE_BC
    saveVarBin(strFileBc, nodeTypeSave, MEM_SIZE_SCALAR, false);
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    saveVarBin(strFileFx, h_BC_Fx, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileFy, h_BC_Fy, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileFz, h_BC_Fz, MEM_SIZE_SCALAR, false);
    #endif
}

void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append)
{
    FILE* outFile = nullptr;
    if(append)
        outFile = fopen(strFile.c_str(), "ab");
    else
        outFile = fopen(strFile.c_str(), "wb");
    if(outFile != nullptr)
    {
        fwrite(var, memSize, 1, outFile);
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strFile.c_str());
    }
}



void folderSetup()
{
// Windows
#if defined(_WIN32)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "\\\\"; // adds "\\"
    strPath += ID_SIM;
    std::string cmd = "md ";
    cmd += strPath;
    system(cmd.c_str());
    return;
#endif // !_WIN32

// Unix
#if defined(__APPLE__) || defined(__MACH__) || defined(__linux__)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "/";
    strPath += ID_SIM;
    std::string cmd = "mkdir -p ";
    cmd += strPath;
    system(cmd.c_str());
    return;
#endif // !Unix
    printf("I don't know how to setup folders for your operational system :(\n");
    return;
}


std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext)
{
    unsigned int n_zeros = 0, pot_10 = 10;
    unsigned int aux1 = 1000000;  // 6 numbers on step
    // calculate number of zeros
    if (step != 0)
        for (n_zeros = 0; step * pot_10 < aux1; pot_10 *= 10)
            n_zeros++;
    else
        n_zeros = 6;

    // generates the file name as "PATH_FILES/id/id_varName000000.bin"
    std::string strFile = PATH_FILES;
    strFile += "/";
    strFile += ID_SIM;
    strFile += "/";
    strFile += ID_SIM;
    strFile += "_";
    strFile += varName;
    for (unsigned int i = 0; i < n_zeros; i++)
        strFile += "0";
    strFile += std::to_string(step);
    strFile += ext;

    return strFile;
}

std::string getSimInfoString(int step,dfloat MLUPS)
{
    std::ostringstream strSimInfo("");
    
    strSimInfo << std::scientific;
    strSimInfo << std::setprecision(6);
    
    strSimInfo << "---------------------------- SIMULATION INFORMATION ----------------------------\n";
    strSimInfo << "      Simulation ID: " << ID_SIM << "\n";
    #ifdef D3Q19
    strSimInfo << "       Velocity set: D3Q19\n";
    #endif // !D3Q19
    #ifdef D3Q27
    strSimInfo << "       Velocity set: D3Q27\n";
    #endif // !D3Q27
    #ifdef SINGLE_PRECISION
        strSimInfo << "          Precision: float\n";
    #else
        strSimInfo << "          Precision: double\n";
    #endif
    strSimInfo << "                 NX: " << NX << "\n";
    strSimInfo << "                 NY: " << NY << "\n";
    strSimInfo << "                 NZ: " << NZ << "\n";
    strSimInfo << "           NZ_TOTAL: " << NZ_TOTAL << "\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "                Tau: " << TAU << "\n";
    strSimInfo << "               Umax: " << U_MAX << "\n";
    strSimInfo << "                 FX: " << FX << "\n";
    strSimInfo << "                 FY: " << FY << "\n";
    strSimInfo << "                 FZ: " << FZ << "\n";
    strSimInfo << "         Save steps: " << MACR_SAVE << "\n";
    strSimInfo << "             Nsteps: " << step << "\n";
    strSimInfo << "              MLUPS: " << MLUPS << "\n";
    strSimInfo << std::scientific << std::setprecision(0);
    strSimInfo << "       Bx x By x Bz: " << BLOCK_NX << "x" << BLOCK_NY << "x"<< BLOCK_NZ << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";

    strSimInfo << "\n------------------------------ BOUNDARY CONDITIONS -----------------------------\n";
    #ifdef BC_MOMENT_BASED
    strSimInfo << "            BC mode: Moment Based \n";
    #endif
    strSimInfo << "            BC type: " << STR(BC_PROBLEM) << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";


    #ifdef NON_NEWTONIAN_FLUID
    strSimInfo << "\n------------------------------ NON NEWTONIAN FLUID -----------------------------\n";
    strSimInfo << std::scientific << std::setprecision(6);
    
    #ifdef POWERLAW
    strSimInfo << "              Model: Power-Law\n";
    strSimInfo << "        Power index: " << N_INDEX << "\n";
    strSimInfo << " Consistency factor: " << K_CONSISTENCY << "\n";
    strSimInfo << "            Gamma 0: " << GAMMA_0 << "\n";
    #endif // POWERLAW

    #ifdef BINGHAM
    strSimInfo << "              Model: Bingham\n";
    strSimInfo << "  Plastic viscosity: " << VISC << "\n";
    strSimInfo << "       Yield stress: " << S_Y << "\n";
    strSimInfo << "      Plastic omega: " << OMEGA_P << "\n";
    #endif // BINGHAM
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif // NON_NEWTONIAN_FLUID
    #ifdef LES_MODEL
    strSimInfo << "\t Smagorisky Constant:" << CONST_SMAGORINSKY <<"\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif //LES
    #ifdef THERMAL_MODEL 
    strSimInfo << "\n------------------------------ THERMAL -----------------------------\n";
        strSimInfo << std::scientific << std::setprecision(2);
    strSimInfo << "     Prandtl Number: " << T_PR_NUMBER << "\n";
        strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << "    Rayleigh Number: " << T_RA_NUMBER << "\n";
    strSimInfo << "     Grashof Number: " << T_GR_NUMBER << "\n";
       strSimInfo << std::scientific << std::setprecision(3);
    strSimInfo << "            Delta T: " << T_DELTA_T << "\n";
    strSimInfo << "        Reference T: " << T_REFERENCE << "\n";
    strSimInfo << "             Cold T: " << T_COLD << "\n";
    strSimInfo << "              Hot T: " << T_HOT << "\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "       Thermal Diff: " << T_DIFFUSIVITY << "\n";
    strSimInfo << "   Grav_t_Exp.Coeff: " << T_gravity_t_beta << "\n";
       strSimInfo << std::scientific << std::setprecision(2);
    strSimInfo << "          Gravity_x: " << gravity_vector[0] << "\n";
    strSimInfo << "          Gravity_y: " << gravity_vector[1] << "\n";
    strSimInfo << "          Gravity_z: " << gravity_vector[2] << "\n";
       strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "              G_TAU: " << G_TAU << "\n";
    strSimInfo << "            G_OMEGA: " << G_OMEGA << "\n";

    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif// THERMAL_MODEL



    return strSimInfo.str();
}

void saveSimInfo(int step,dfloat MLUPS)
{
    std::string strInf = PATH_FILES;
    strInf += "/";
    strInf += ID_SIM;
    strInf += "/";
    strInf += ID_SIM;
    strInf += "_info.txt"; // generate file name (with path)
    FILE* outFile = nullptr;

    outFile = fopen(strInf.c_str(), "w");
    if(outFile != nullptr)
    {
        std::string strSimInfo = getSimInfoString(step,MLUPS);
        fprintf(outFile, strSimInfo.c_str());
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strInf.c_str());
    }
    
}
/**/


void saveTreatData(std::string fileName, std::string dataString, int step)
{
    #if SAVEDATA
    std::string strInf = PATH_FILES;
    strInf += "/";
    strInf += ID_SIM;
    strInf += "/";
    strInf += ID_SIM;
    strInf += fileName;
    strInf += ".txt"; // generate file name (with path)
    std::ifstream file(strInf.c_str());
    std::ofstream outfile;

    if(step == MACR_SAVE){ //check if first time step to save data
        outfile.open(strInf.c_str());
    }else{
        if (file.good()) {
            outfile.open(strInf.c_str(), std::ios::app);
        }else{ 
            outfile.open(strInf.c_str());
        }
    }


    outfile << dataString.c_str() << std::endl; 
    outfile.close(); 
    #endif
    #if CONSOLEPRINT
    printf("%s \n",dataString.c_str());
    #endif
}