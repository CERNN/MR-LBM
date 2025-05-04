totalKineticEnergy(fMom,step);
#ifdef CONVECTION_DIFFUSION_TRANSPORT
totalSpringEnergy(fMom,step);
#endif
/*

checkCudaErrors(cudaDeviceSynchronize());
checkCudaErrors(cudaMemcpy(h_fMom, fMom, sizeof(dfloat) * NUMBER_LBM_NODES*NUMBER_MOMENTS, cudaMemcpyDeviceToHost));
checkCudaErrors(cudaDeviceSynchronize());

dfloat ux, uy, uz;
dfloat Axx,Ayy,Azz;

dfloat E_kinetic = 0.0;
dfloat E_spring = 0.0;
dfloat A_spring = 0.0;

dfloat Pixx, Piyy, Pizz;



for (int z = 0; z <NZ_TOTAL; z++){
    for(int y = 0; y< NY;y++){
        for (int x = 0; x <NX; x++){
            ux = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            uy = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            uz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

            E_kinetic += (ux*ux + uy*uy + uz*uz);

            Axx = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO; 
            Ayy = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO; 
            Azz = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_ZZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO; 

            Pixx = nu_p*inv_lambda * (Axx - 1.0);
            Piyy = nu_p*inv_lambda * (Ayy - 1.0);
            Pizz = nu_p*inv_lambda * (Azz - 1.0);

            A_spring += (Axx + Ayy + Azz);
            E_spring += (Pixx + Piyy + Pizz);

            if(x == 0 && y == 0 && z==0)
                printf("step %d Axx %f Ayy %f Azz %f Pixx %f Piyy %f Pizz %f \n",step,Axx,Ayy,Azz,Pixx,Piyy,Pizz);
        }
    }
}


E_kinetic = (E_kinetic/2.0)/(U_MAX*U_MAX*NUMBER_LBM_NODES);
A_spring = (A_spring/2.0)/(U_MAX*U_MAX*NUMBER_LBM_NODES);
E_spring = (E_spring/2.0)/(U_MAX*U_MAX*NUMBER_LBM_NODES);



std::ostringstream strDataInfo("");
strDataInfo << std::scientific;
strDataInfo << std::setprecision(6);

strDataInfo <<"step,"<< step << "," << E_kinetic << "," << A_spring << "," << E_spring;
saveTreatData("_Energy_Info",strDataInfo.str(),step);
*/