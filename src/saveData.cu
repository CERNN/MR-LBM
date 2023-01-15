#include "saveData.cuh"


__host__
void probeExport(
        dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    #ifdef NON_NEWTONIAN_FLUID
    dfloat* omega,
    #endif
    unsigned int step
){
    /*
    int x_0,x_1,x_2,x_3,x_4;
    int y_0,y_1,y_2,y_3,y_4;
    int z_0;
    x_0 = probe_x;
    y_0 = probe_y;
    z_0 = probe_z;

    x_1 = (NX/4);
    x_2 = (NX/4);
    x_3 = (3*NX/4);
    x_4 = (3*NX/4);

    y_1 = (NY/4);
    y_2 = (3*NY/4);
    y_3 = (3*NY/4);
    y_4 = (NY/4);

    dfloat p_ux_0,p_uy_0,p_uz_0,p_rho_0;

    dfloat p_ux_1,p_uy_1;
    dfloat p_ux_2,p_uy_2;
    dfloat p_ux_3,p_uy_3;
    dfloat p_ux_4,p_uy_4;

    p_rho_0 = h_fMom[idxMom(x_0%BLOCK_NX, y_0%BLOCK_NY, z_0%BLOCK_NZ, 0, x_0/BLOCK_NX, y_0/BLOCK_NY, z_0/BLOCK_NZ)];
    p_ux_0  = h_fMom[idxMom(x_0%BLOCK_NX, y_0%BLOCK_NY, z_0%BLOCK_NZ, 1, x_0/BLOCK_NX, y_0/BLOCK_NY, z_0/BLOCK_NZ)];
    p_uy_0  = h_fMom[idxMom(x_0%BLOCK_NX, y_0%BLOCK_NY, z_0%BLOCK_NZ, 2, x_0/BLOCK_NX, y_0/BLOCK_NY, z_0/BLOCK_NZ)];
    p_uz_0  = h_fMom[idxMom(x_0%BLOCK_NX, y_0%BLOCK_NY, z_0%BLOCK_NZ, 3, x_0/BLOCK_NX, y_0/BLOCK_NY, z_0/BLOCK_NZ)];

    p_ux_1  = h_fMom[idxMom(x_1%BLOCK_NX, y_1%BLOCK_NY, z_0%BLOCK_NZ, 1, x_1/BLOCK_NX, y_1/BLOCK_NY, z_0/BLOCK_NZ)];
    p_uy_1  = h_fMom[idxMom(x_1%BLOCK_NX, y_1%BLOCK_NY, z_0%BLOCK_NZ, 2, x_1/BLOCK_NX, y_1/BLOCK_NY, z_0/BLOCK_NZ)];
    p_ux_2  = h_fMom[idxMom(x_2%BLOCK_NX, y_2%BLOCK_NY, z_0%BLOCK_NZ, 1, x_2/BLOCK_NX, y_2/BLOCK_NY, z_0/BLOCK_NZ)];
    p_uy_2  = h_fMom[idxMom(x_2%BLOCK_NX, y_2%BLOCK_NY, z_0%BLOCK_NZ, 2, x_2/BLOCK_NX, y_2/BLOCK_NY, z_0/BLOCK_NZ)];
    p_ux_3  = h_fMom[idxMom(x_3%BLOCK_NX, y_3%BLOCK_NY, z_0%BLOCK_NZ, 1, x_3/BLOCK_NX, y_3/BLOCK_NY, z_0/BLOCK_NZ)];
    p_uy_3  = h_fMom[idxMom(x_3%BLOCK_NX, y_3%BLOCK_NY, z_0%BLOCK_NZ, 2, x_3/BLOCK_NX, y_3/BLOCK_NY, z_0/BLOCK_NZ)];
    p_ux_4  = h_fMom[idxMom(x_4%BLOCK_NX, y_4%BLOCK_NY, z_0%BLOCK_NZ, 1, x_4/BLOCK_NX, y_4/BLOCK_NY, z_0/BLOCK_NZ)];
    p_uy_4  = h_fMom[idxMom(x_4%BLOCK_NX, y_4%BLOCK_NY, z_0%BLOCK_NZ, 2, x_4/BLOCK_NX, y_4/BLOCK_NY, z_0/BLOCK_NZ)];


    
   printf("%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\t%0.7e\n",p_rho_0,p_ux_0,p_uy_0,p_uz_0,p_ux_1,p_uy_1,p_ux_2,p_uy_2,p_ux_3,p_uy_3,p_ux_4,p_uy_4);

*/




    dfloat t_ux0, t_ux1;
    dfloat dy_ux = 0.0;
    int y0 = NY-1;
    int y1 = NY-2;
    int count = 0;

    //right side of the equation 10
    for (int z = 1 ; z <NZ_TOTAL-1 ; z++){
        for (int x = 1; x< NX-1;x++){
            t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y0%BLOCK_NY, z%BLOCK_NZ, 1, x/BLOCK_NX, y0/BLOCK_NY, z/BLOCK_NZ)];
            t_ux1 = h_fMom[idxMom(x%BLOCK_NX, y1%BLOCK_NY, z%BLOCK_NZ, 1, x/BLOCK_NX, y1/BLOCK_NY, z/BLOCK_NZ)];

            dy_ux += (t_ux0-t_ux1)*(t_ux0-t_ux1);
            count++;
        }
    }
    dfloat LS = dy_ux/count;
    LS /= 4*N;



    dfloat t_uy0,t_uz0;
    dfloat t_mxx0,t_mxy0,t_mxz0,t_myy0,t_myz0,t_mzz0;
    dfloat Sxx = 0;
    dfloat Sxy = 0;
    dfloat Sxz = 0;
    dfloat Syy = 0;
    dfloat Syz = 0;
    dfloat Szz = 0;
    dfloat SS = 0;

    //left side of the equation
    for (int z = 0 ; z <NZ_TOTAL; z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                t_ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 1, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 2, x/BLOCK_NX, y0/BLOCK_NY, z/BLOCK_NZ)];
                t_uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 3, x/BLOCK_NX, y0/BLOCK_NY, z/BLOCK_NZ)];

                t_mxx0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 4, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mxy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 5, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mxz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 9, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_myy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 7, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_myz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 8, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                t_mzz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 9, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];



                Sxx = (as2/(2*TAU))*(t_ux0*t_ux0-t_mxx0);
                Sxy = (as2/(2*TAU))*(t_ux0*t_uy0-t_mxy0);
                Sxz = (as2/(2*TAU))*(t_ux0*t_uz0-t_mxz0);
                Syy = (as2/(2*TAU))*(t_uy0*t_uy0-t_myy0);
                Syz = (as2/(2*TAU))*(t_uy0*t_uz0-t_myz0);
                Szz = (as2/(2*TAU))*(t_uz0*t_uz0-t_mzz0);
                SS += ( Sxx * Sxx + Syy * Syy + Szz * Szz + 2*(Sxy * Sxy + Sxz * Sxz +  Syz * Syz)) ;

            }
        }
    }

    SS = SS / (NUMBER_LBM_NODES);




    printf("%0.7e\t%0.7e\t%0.7e\n",LS,SS,SS/LS);


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
    unsigned int step
){
    size_t indexMacr;

    dfloat meanRho;
    meanRho  =  0;

    for(int z = 0; z< NZ;z++){
        ///printf("z %d \n", z);
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                indexMacr = idxScalarGlobal(x,y,z);

                rho[indexMacr] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 0, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                ux[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 1, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 2, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 3, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #ifdef NON_NEWTONIAN_FLUID
                omega[indexMacr] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 10, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]; 
                #endif
                //data += rho[indexMacr]*(ux[indexMacr]*ux[indexMacr] + uy[indexMacr]*uy[indexMacr] + uz[indexMacr]*uz[indexMacr]);
                meanRho += rho[indexMacr];
            }
        }
    }
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
    unsigned int nSteps
){
// Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz, strFileOmega;

    strFileRho = getVarFilename("rho", nSteps, ".bin");
    strFileUx = getVarFilename("ux", nSteps, ".bin");
    strFileUy = getVarFilename("uy", nSteps, ".bin");
    strFileUz = getVarFilename("uz", nSteps, ".bin");
    #ifdef NON_NEWTONIAN_FLUID
    strFileOmega = getVarFilename("omega", nSteps, ".bin");
    #endif
    // saving files
    saveVarBin(strFileRho, rho, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUx, ux, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUy, uy, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUz, uz, MEM_SIZE_SCALAR, false);
    #ifdef NON_NEWTONIAN_FLUID
    saveVarBin(strFileOmega, omega, MEM_SIZE_SCALAR, false);
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

std::string getSimInfoString(int step)
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
    strSimInfo << "--------------------------------------------------------------------------------\n";

    return strSimInfo.str();
}

void saveSimInfo(int step)
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
        std::string strSimInfo = getSimInfoString(step);
        fprintf(outFile, strSimInfo.c_str());
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strInf.c_str());
    }
    
}