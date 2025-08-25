//copy full macroscopic field
checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());


dfloat t_ux0, t_uy0,t_uz0;
dfloat m_ux = 0.0;
dfloat m_uy = 0.0;
dfloat m_uz = 0.0;



//temporary variables
dfloat ux_mean[NY/2];
dfloat uy_mean[NY/2];
dfloat uz_mean[NY/2];
int y_wall;

dfloat ux_fluct[NY/2];
dfloat uy_fluct[NY/2];
dfloat uz_fluct[NY/2];
dfloat uxuy_fluct[NY/2];
dfloat uxuz_fluct[NY/2];
dfloat uyuz_fluct[NY/2];


//Update the mean field values
dfloat mean_counter = 1.0/((dfloat)(step/MACR_SAVE)+1.0);


for(int y = 0; y< NY;y++){
    //distance to any wall
    y_wall = (y <= (NY-1)/2) ? y : (NY-1 - y);
    
    ux_mean[y_wall] = 0;
    uy_mean[y_wall] = 0;
    uz_mean[y_wall] = 0;

    ux_fluct[y_wall] = 0;
    uy_fluct[y_wall] = 0;
    uz_fluct[y_wall] = 0;
    uxuy_fluct[y_wall] = 0;
    uxuz_fluct[y_wall] = 0;
    uyuz_fluct[y_wall] = 0;



    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int x = 0; x< NX;x++){
            //current lattice value
            t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            t_uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            t_uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

            //retrive mean values
            m_ux = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            m_uy = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            m_uz = fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

            //update mean values
            m_ux = m_ux + (t_ux0 - m_ux)*(mean_counter);
            m_uy = m_uy + (t_uy0 - m_uy)*(mean_counter);
            m_uz = m_uz + (t_uz0 - m_uz)*(mean_counter);

            fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_ux;
            fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uy;
            fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uz;

            ux_mean[y_wall] += m_ux/(NX*NZ_TOTAL);
            uy_mean[y_wall] += m_uy/(NX*NZ_TOTAL);
            uz_mean[y_wall] += m_uz/(NX*NZ_TOTAL);

            ux_fluct[y_wall] += (t_ux0 - m_ux)*(t_ux0 - m_ux)/(NX*NZ_TOTAL);
            uy_fluct[y_wall] += (t_uy0 - m_uy)*(t_uy0 - m_uy)/(NX*NZ_TOTAL);
            uz_fluct[y_wall] += (t_uz0 - m_uz)*(t_uz0 - m_uz)/(NX*NZ_TOTAL);

            uxuy_fluct[y_wall] += (t_ux0 - m_ux)*(t_uy0 - m_uy)/(NX*NZ_TOTAL);
            uxuz_fluct[y_wall] += (t_ux0 - m_ux)*(t_uz0 - m_uz)/(NX*NZ_TOTAL);
            uyuz_fluct[y_wall] += (t_uy0 - m_uy)*(t_uz0 - m_uz)/(NX*NZ_TOTAL);

        }
    }
}


//set string for export
std::ostringstream strDataInfo_ux_mean("");
std::ostringstream strDataInfo_uy_mean("");
std::ostringstream strDataInfo_uz_mean("");
std::ostringstream strDataInfo_ux_fluct("");
std::ostringstream strDataInfo_uy_fluct("");
std::ostringstream strDataInfo_uz_fluct("");
std::ostringstream strDataInfo_uxuy_fluct("");
std::ostringstream strDataInfo_uxuz_fluct("");
std::ostringstream strDataInfo_uyuz_fluct("");

strDataInfo_ux_mean <<"step,"<< step;
strDataInfo_uy_mean <<"step,"<< step;
strDataInfo_uz_mean <<"step,"<< step;
strDataInfo_ux_fluct <<"step,"<< step;
strDataInfo_uy_fluct <<"step,"<< step;
strDataInfo_uz_fluct <<"step,"<< step;
strDataInfo_uxuy_fluct <<"step,"<< step;
strDataInfo_uxuz_fluct <<"step,"<< step;
strDataInfo_uyuz_fluct <<"step,"<< step;

for(int y = 0; y< NY/2;y++){
    strDataInfo_ux_mean     << "," << ux_mean[y];
    strDataInfo_uy_mean     << "," << uy_mean[y];
    strDataInfo_uz_mean     << "," << uz_mean[y];
    strDataInfo_ux_fluct    << "," << ux_fluct[y];
    strDataInfo_uy_fluct    << "," << uy_fluct[y];
    strDataInfo_uz_fluct    << "," << uz_fluct[y];
    strDataInfo_uxuy_fluct  << "," << uxuy_fluct[y];
    strDataInfo_uxuz_fluct  << "," << uxuz_fluct[y];
    strDataInfo_uyuz_fluct  << "," << uyuz_fluct[y];
}

saveTreatData("_turbulent_ux_mean",strDataInfo_ux_mean.str(),step);
saveTreatData("_turbulent_uy_mean",strDataInfo_uy_mean.str(),step);
saveTreatData("_turbulent_uz_mean",strDataInfo_uz_mean.str(),step);
saveTreatData("_turbulent_ux_fluct",strDataInfo_ux_fluct.str(),step);
saveTreatData("_turbulent_uy_fluct",strDataInfo_uy_fluct.str(),step);
saveTreatData("_turbulent_uz_fluct",strDataInfo_uz_fluct.str(),step);
saveTreatData("_turbulent_uxuy_fluct",strDataInfo_uxuy_fluct.str(),step);
saveTreatData("_turbulent_uxuz_fluct",strDataInfo_uxuz_fluct.str(),step);
saveTreatData("_turbulent_uyuz_fluct",strDataInfo_uyuz_fluct.str(),step);


