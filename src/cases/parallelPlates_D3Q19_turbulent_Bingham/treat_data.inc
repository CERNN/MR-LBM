//copy full macroscopic field
checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());


dfloat t_ux0, t_uy0,t_uz0;
dfloat m_ux = 0.0;
dfloat m_uy = 0.0;
dfloat m_uz = 0.0;



//temporary variables
//mean flow
dfloat ux_mean[NY/2];
dfloat uy_mean[NY/2];
dfloat uz_mean[NY/2];
int y_wall;

//fluctuations
dfloat ux_fluct[NY/2];
dfloat uy_fluct[NY/2];
dfloat uz_fluct[NY/2];
dfloat uxuy_fluct[NY/2];
dfloat uxuz_fluct[NY/2];
dfloat uyuz_fluct[NY/2];

// higher-order moments
dfloat ux_skew_accum[NY/2];
dfloat uy_skew_accum[NY/2];
dfloat uz_skew_accum[NY/2];

dfloat ux_kurt_accum[NY/2];
dfloat uy_kurt_accum[NY/2];
dfloat uz_kurt_accum[NY/2];

dfloat ux_skew[NY/2];
dfloat uy_skew[NY/2];
dfloat uz_skew[NY/2];

dfloat ux_kurt[NY/2];
dfloat uy_kurt[NY/2];
dfloat uz_kurt[NY/2];

//kinetic energy and invariants
dfloat turb_kinetic[NY/2];
dfloat invariant2[NY/2];
dfloat invariant3[NY/2];
dfloat R[3][3];
dfloat aij[3][3];



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

    turb_kinetic[y_wall] = 0;
    invariant2[y_wall] = 0;
    invariant3[y_wall] = 0;

    ux_skew_accum[y_wall] = 0;
    uy_skew_accum[y_wall] = 0;
    uz_skew_accum[y_wall] = 0;

    ux_kurt_accum[y_wall] = 0;
    uy_kurt_accum[y_wall] = 0;
    uz_kurt_accum[y_wall] = 0;



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

            //local flucuation
            dfloat ux_fl = t_ux0 - m_ux;
            dfloat uy_fl = t_uy0 - m_uy;
            dfloat uz_fl = t_uz0 - m_uz;

            //update mean values
            m_ux = m_ux + (ux_fl)*(mean_counter);
            m_uy = m_uy + (uy_fl)*(mean_counter);
            m_uz = m_uz + (uz_fl)*(mean_counter);

            fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_ux;
            fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uy;
            fMom_mean[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = m_uz;

            ux_mean[y_wall] += m_ux/(NX*NZ_TOTAL);
            uy_mean[y_wall] += m_uy/(NX*NZ_TOTAL);
            uz_mean[y_wall] += m_uz/(NX*NZ_TOTAL);


            ux_fluct[y_wall] += (ux_fl)*(ux_fl)/(NX*NZ_TOTAL);
            uy_fluct[y_wall] += (uy_fl)*(uy_fl)/(NX*NZ_TOTAL);
            uz_fluct[y_wall] += (uz_fl)*(uz_fl)/(NX*NZ_TOTAL);

            uxuy_fluct[y_wall] += (ux_fl)*(uy_fl)/(NX*NZ_TOTAL);
            uxuz_fluct[y_wall] += (ux_fl)*(uz_fl)/(NX*NZ_TOTAL);
            uyuz_fluct[y_wall] += (uy_fl)*(uz_fl)/(NX*NZ_TOTAL);

            // third- and fourth-order moments
            ux_skew_accum[y_wall] += (ux_fl*ux_fl*ux_fl)/(NX*NZ_TOTAL);
            uy_skew_accum[y_wall] += (uy_fl*uy_fl*uy_fl)/(NX*NZ_TOTAL);
            uz_skew_accum[y_wall] += (uz_fl*uz_fl*uz_fl)/(NX*NZ_TOTAL);

            ux_kurt_accum[y_wall] += (ux_fl*ux_fl*ux_fl*ux_fl)/(NX*NZ_TOTAL);
            uy_kurt_accum[y_wall] += (uy_fl*uy_fl*uy_fl*uy_fl)/(NX*NZ_TOTAL);
            uz_kurt_accum[y_wall] += (uz_fl*uz_fl*uz_fl*uz_fl)/(NX*NZ_TOTAL);

        }
    }


    dfloat sigma_x = sqrt(ux_fluct[y_wall]);
    dfloat sigma_y = sqrt(uy_fluct[y_wall]);
    dfloat sigma_z = sqrt(uz_fluct[y_wall]);

    ux_skew[y_wall] = (sigma_x > 1e-14) ? ux_skew_accum[y_wall] / (sigma_x*sigma_x*sigma_x) : 0.0;
    uy_skew[y_wall] = (sigma_y > 1e-14) ? uy_skew_accum[y_wall] / (sigma_y*sigma_y*sigma_y) : 0.0;
    uz_skew[y_wall] = (sigma_z > 1e-14) ? uz_skew_accum[y_wall] / (sigma_z*sigma_z*sigma_z) : 0.0;

    ux_kurt[y_wall] = (sigma_x > 1e-14) ? ux_kurt_accum[y_wall] / (sigma_x*sigma_x*sigma_x*sigma_x) : 0.0;
    uy_kurt[y_wall] = (sigma_y > 1e-14) ? uy_kurt_accum[y_wall] / (sigma_y*sigma_y*sigma_y*sigma_y) : 0.0;
    uz_kurt[y_wall] = (sigma_z > 1e-14) ? uz_kurt_accum[y_wall] / (sigma_z*sigma_z*sigma_z*sigma_z) : 0.0;



    // Build Reynolds stress tensor R_ij
    R[0][0] = ux_fluct[y_wall];      // <u'x u'x>
    R[1][1] = uy_fluct[y_wall];      // <u'y u'y>
    R[2][2] = uz_fluct[y_wall];      // <u'z u'z>
    R[0][1] = R[1][0] = uxuy_fluct[y_wall];
    R[0][2] = R[2][0] = uxuz_fluct[y_wall];
    R[1][2] = R[2][1] = uyuz_fluct[y_wall];

    // Turbulent kinetic energy
    turb_kinetic[y_wall] = 0.5 * (R[0][0] + R[1][1] + R[2][2]);

    // Anisotropy tensor a_ij = R_ij/(2k) - delta_ij/3
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            aij[i][j] = R[i][j] / (2.0 * turb_kinetic[y_wall]);
            if (i == j) aij[i][j] -= 1.0/3.0;
        }
    }

    // Invariant II = -0.5 * a_ij a_ji
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            invariant2[y_wall] += aij[i][j] * aij[j][i];
        }
    }
    invariant2[y_wall] *= -0.5;

    invariant3[y_wall] = aij[0][0]*(aij[1][1]*aij[2][2] - aij[1][2]*aij[2][1]) -
                        aij[0][1]*(aij[1][0]*aij[2][2] - aij[1][2]*aij[2][0]) +
                        aij[0][2]*(aij[1][0]*aij[2][1] - aij[1][1]*aij[2][0]);


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

std::ostringstream strDataInfo_k("");
std::ostringstream strDataInfo_II("");
std::ostringstream strDataInfo_III("");

std::ostringstream strDataInfo_ux_skew("");
std::ostringstream strDataInfo_uy_skew("");
std::ostringstream strDataInfo_uz_skew("");
std::ostringstream strDataInfo_ux_kurt("");
std::ostringstream strDataInfo_uy_kurt("");
std::ostringstream strDataInfo_uz_kurt("");

strDataInfo_ux_mean <<"step,"<< step;
strDataInfo_uy_mean <<"step,"<< step;
strDataInfo_uz_mean <<"step,"<< step;
strDataInfo_ux_fluct <<"step,"<< step;
strDataInfo_uy_fluct <<"step,"<< step;
strDataInfo_uz_fluct <<"step,"<< step;
strDataInfo_uxuy_fluct <<"step,"<< step;
strDataInfo_uxuz_fluct <<"step,"<< step;
strDataInfo_uyuz_fluct <<"step,"<< step;

strDataInfo_k  << "step," << step;
strDataInfo_II << "step," << step;
strDataInfo_III<< "step," << step;

strDataInfo_ux_skew <<"step,"<< step;
strDataInfo_uy_skew <<"step,"<< step;
strDataInfo_uz_skew <<"step,"<< step;
strDataInfo_ux_kurt <<"step,"<< step;
strDataInfo_uy_kurt <<"step,"<< step;
strDataInfo_uz_kurt <<"step,"<< step;

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
    strDataInfo_k  << "," << turb_kinetic[y];
    strDataInfo_II << "," << invariant2[y];
    strDataInfo_III<< "," << invariant3[y];
    strDataInfo_ux_skew << "," << ux_skew[y];
    strDataInfo_uy_skew << "," << uy_skew[y];
    strDataInfo_uz_skew << "," << uz_skew[y];
    strDataInfo_ux_kurt << "," << ux_kurt[y];
    strDataInfo_uy_kurt << "," << uy_kurt[y];
    strDataInfo_uz_kurt << "," << uz_kurt[y];
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
saveTreatData("_turbulent_k",   strDataInfo_k.str(), step);
saveTreatData("_turbulent_II",  strDataInfo_II.str(), step);
saveTreatData("_turbulent_III", strDataInfo_III.str(), step);
saveTreatData("_turbulent_ux_skew",  strDataInfo_ux_skew.str(), step);
saveTreatData("_turbulent_uy_skew",  strDataInfo_uy_skew.str(), step);
saveTreatData("_turbulent_uz_skew",  strDataInfo_uz_skew.str(), step);
saveTreatData("_turbulent_ux_kurt",  strDataInfo_ux_kurt.str(), step);
saveTreatData("_turbulent_uy_kurt",  strDataInfo_uy_kurt.str(), step);
saveTreatData("_turbulent_uz_kurt",  strDataInfo_uz_kurt.str(), step);


