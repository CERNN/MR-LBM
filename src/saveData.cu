#include "saveData.cuh"

__host__
void linearMacr(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    unsigned int step
){
    size_t indexMacr;

    dfloat rho0 =0;
    dfloat ux0  =0;
    dfloat uy0  =0;
    dfloat uz0  =0;

    dfloat u[NY];
    dfloat v[NY];
    dfloat w[NY];

    dfloat p[NY];

    dfloat uu[NY];
    dfloat vv[NY];
    dfloat ww[NY];
    dfloat uv[NY];
    dfloat uw[NY];
    dfloat vw[NY];
    dfloat pp[NY];

    dfloat loc_p = 0;

    dfloat rho_avg_inst = 0.0;

    for(int y = 0; y< NY;y++){
        for(int z = 0; z< NZ;z++){
            for(int x = 0; x< NX;x++){
                rho_avg_inst += h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 0, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
            }
        }
    }
    rho_avg_inst /= NX*NY*NZ;

    for(int y = 0; y< NY;y++){
        u[y] = 0.0;
        v[y] = 0.0;
        w[y] = 0.0;

        p[y] = 0.0;

        uu[y] = 0.0;
        vv[y] = 0.0;
        ww[y] = 0.0;
        uv[y] = 0.0;
        uw[y] = 0.0;
        vw[y] = 0.0;
        pp[y] = 0.0;


        for(int z = 0; z< NZ;z++){
            for(int x = 0; x< NX;x++){
                indexMacr = idxScalarGlobal(x,y,z);

                rho0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 0, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                ux0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 1, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uy0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 2, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uz0 = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, 3, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                rho[indexMacr] =rho0;
                ux[indexMacr] = ux0;
                uy[indexMacr] = uy0;
                uz[indexMacr] = uz0;

                loc_p = (1.0/3.0)*rho0 - (1.0/3.0)*rho_avg_inst;
                
                u[y] += ux0;
                v[y] += uy0;
                w[y] += uz0;
                p[y] += loc_p;
                uu[y] += ux0*ux0;
                vv[y] += uy0*uy0;
                ww[y] += uz0*uz0;
                uv[y] += ux0*uy0;
                uw[y] += ux0*uz0;
                vw[y] += uy0*uz0;
                pp[y] += loc_p*loc_p;
            }
        }
        u[y] /= NX*NZ;
        v[y] /= NX*NZ;
        w[y] /= NX*NZ;
        p[y] /= NX*NZ;
        uu[y] /= NX*NZ;
        vv[y] /= NX*NZ;
        ww[y] /= NX*NZ;
        uv[y] /= NX*NZ;
        uw[y] /= NX*NZ;
        vw[y] /= NX*NZ;
        pp[y] /= NX*NZ;
    }

    
    
    printf(" %d 00 ",step);
    for(int y = 0; y< NY;y++)
        printf("%0.7e ",u[y]);
    
    printf("\n %d 01 ",step);
    for(int y = 0; y< NY;y++)
        printf("%0.7e ",u[y]);

    printf("\n %d 02 ",step);
    for(int y = 0; y< NY;y++)
        printf("%0.7e ",v[y]);

    printf("\n %d 03 ",step);
    for(int y = 0; y< NY;y++)
        printf("%0.7e ",w[y]);




    printf("\n %d 11 ",step);
    for(int y = 0; y< NY;y++)
        printf("%0.7e ", uu[y]);

    printf("\n %d 12 ",step);        
    for(int y = 0; y< NY;y++)
        printf("%0.7e ", uv[y]);

    printf("\n %d 13 ",step);
    for(int y = 0; y< NY;y++)        
        printf("%0.7e ", uw[y]);

    printf("\n %d 22 ",step);
    for(int y = 0; y< NY;y++)        
        printf("%0.7e ", vv[y]);

    printf("\n %d 23 ",step);     
    for(int y = 0; y< NY;y++)        
        printf("%0.7e ", vw[y]);

    printf("\n %d 33 ",step);
    for(int y = 0; y< NY;y++)        
        printf("%0.7e ", ww[y]);
        

    printf("\n");
    

    //Ekin = Ekin/(2*N*N*N);
    //printf("%0.7e \n",Ekin);
}


__host__
void saveMacr(
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    unsigned int nSteps
){
// Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz;

    strFileRho = getVarFilename("rho", nSteps, ".bin");
    strFileUx = getVarFilename("ux", nSteps, ".bin");
    strFileUy = getVarFilename("uy", nSteps, ".bin");
    strFileUz = getVarFilename("uz", nSteps, ".bin");

    // saving files
    saveVarBin(strFileRho, rho, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUx, ux, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUy, uy, MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUz, uz, MEM_SIZE_SCALAR, false);
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