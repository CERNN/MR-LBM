#include "saveData.cuh"
#ifdef OMEGA_FIELD
#include "nnf.h"
#endif

__host__
void saveMacr(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif
    #ifdef A_XX_DIST 
    dfloat* Axx,
    #endif
    #ifdef A_XY_DIST 
    dfloat* Axy,
    #endif
    #ifdef A_XZ_DIST 
    dfloat* Axz,
    #endif
    #ifdef A_YY_DIST 
    dfloat* Ayy,
    #endif
    #ifdef A_YZ_DIST 
    dfloat* Ayz,
    #endif
    #ifdef A_ZZ_DIST 
    dfloat* Azz,
    #endif
    #ifdef LOG_CONFORMATION
        #ifdef A_XX_DIST
        dfloat* Cxx,
        #endif
        #ifdef A_XY_DIST
        dfloat* Cxy,
        #endif
        #ifdef A_XZ_DIST
        dfloat* Cxz,
        #endif
        #ifdef A_YY_DIST
        dfloat* Cyy,
        #endif
        #ifdef A_YZ_DIST
        dfloat* Cyz,
        #endif
        #ifdef A_ZZ_DIST
        dfloat* Czz,
        #endif
    #endif //LOG_CONFORMATION
    NODE_TYPE_SAVE_PARAMS_DECLARATION
    BC_FORCES_PARAMS_DECLARATION(h_) 
    unsigned int nSteps
){


    //linearize
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

                #ifdef OMEGA_FIELD
                omega[indexMacr] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]; 
                #endif

                #ifdef SECOND_DIST 
                C[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif
                #ifdef A_XX_DIST 
                Axx[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif
                #ifdef A_XY_DIST 
                Axy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif
                #ifdef A_XZ_DIST 
                Axz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif
                #ifdef A_YY_DIST 
                Ayy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif
                #ifdef A_YZ_DIST 
                Ayz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif
                #ifdef A_ZZ_DIST 
                Azz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_ZZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif
                
                #if NODE_TYPE_SAVE
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


    // Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz;
    std::string strFileOmega;
    std::string strFileC;
    std::string strFileBc; 
    std::string strFileFx, strFileFy, strFileFz;
    std::string strFileAxx, strFileAxy, strFileAxz, strFileAyy, strFileAyz, strFileAzz;

    strFileRho = getVarFilename("rho", nSteps, ".bin");
    strFileUx = getVarFilename("ux", nSteps, ".bin");
    strFileUy = getVarFilename("uy", nSteps, ".bin");
    strFileUz = getVarFilename("uz", nSteps, ".bin");
    #ifdef OMEGA_FIELD
    strFileOmega = getVarFilename("omega", nSteps, ".bin");
    #endif
    #ifdef SECOND_DIST 
    strFileC = getVarFilename("C", nSteps, ".bin");
    #endif
    #ifdef A_XX_DIST 
    strFileAxx = getVarFilename("Axx", nSteps, ".bin");
    #endif
    #ifdef A_XY_DIST 
    strFileAxy = getVarFilename("Axy", nSteps, ".bin");
    #endif
    #ifdef A_XZ_DIST 
    strFileAxz = getVarFilename("Axz", nSteps, ".bin");
    #endif
    #ifdef A_YY_DIST 
    strFileAyy = getVarFilename("Ayy", nSteps, ".bin");
    #endif
    #ifdef A_YZ_DIST 
    strFileAyz = getVarFilename("Ayz", nSteps, ".bin");
    #endif
    #ifdef A_ZZ_DIST 
    strFileAzz = getVarFilename("Azz", nSteps, ".bin");
    #endif
    #if NODE_TYPE_SAVE
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
    #ifdef OMEGA_FIELD
    saveVarBin(strFileOmega, omega, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef SECOND_DIST
    saveVarBin(strFileC, C, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef A_XX_DIST 
    saveVarBin(strFileAxx, Axx, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef A_XY_DIST 
    saveVarBin(strFileAxy, Axy, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef A_XZ_DIST 
    saveVarBin(strFileAxz, Axz, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef A_YY_DIST 
    saveVarBin(strFileAyy, Ayy, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef A_YZ_DIST 
    saveVarBin(strFileAyz, Ayz, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef A_ZZ_DIST 
    saveVarBin(strFileAzz, Azz, MEM_SIZE_SCALAR, false);
    #endif
    
    #if NODE_TYPE_SAVE
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
    strSimInfo << "                 BX: " << BLOCK_NX << "\n";
    strSimInfo << "                 BY: " << BLOCK_NY << "\n";
    strSimInfo << "                 BZ: " << BLOCK_NZ << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";

    strSimInfo << "\n------------------------------ BOUNDARY CONDITIONS -----------------------------\n";
    #ifdef BC_MOMENT_BASED
    strSimInfo << "            BC mode: Moment Based \n";
    #endif
    strSimInfo << "            BC type: " << STR(BC_PROBLEM) << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";


    #ifdef OMEGA_FIELD
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
    #endif // OMEGA_FIELD
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
    #ifdef FENE_P 
    strSimInfo << "\n------------------------------ VISCOELASTIC -----------------------------\n";
        strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << " Weissenberg Number: " << Weissenberg_number << "\n";
    strSimInfo << "    Sum Viscosities: " << SUM_VISC << "\n";
    strSimInfo << "    Viscosity Ratio: " << BETA << "\n";
    strSimInfo << "  Solvent Viscosity: " << VISC << "\n";
    strSimInfo << "  Polymer Viscosity: " << nu_p << "\n";
    strSimInfo << "             Lambda: " << LAMBDA << "\n";
    strSimInfo << "           FENE-P A: " << 1 << "\n"; //todo fix when fenep
    strSimInfo << "           FENE-P B: " << 1 << "\n"; //todo fix when fenep
    strSimInfo << "          FENE-P Re: " << fenep_re << "\n";
    strSimInfo << "\n                                                                         \n";
        strSimInfo << std::scientific << std::setprecision(4);
    strSimInfo << "  Diffusivity ratio: " << CONF_DIFFUSIVITY_RATIO << "\n";
    strSimInfo << "  Diffusivity Coef.: " << CONF_DIFFUSIVITY << "\n";
    strSimInfo << "Conformation Offset: " << CONF_ZERO << "\n";
    strSimInfo << "           CONF_TAU: " << CONF_TAU << "\n";
    strSimInfo << "         CONF_OMEGA: " << CONF_OMEGA << "\n";
    strSimInfo << "     CONF_DIFF_FLUC: " << CONF_DIFF_FLUC << "\n";
    strSimInfo << "           CONF_AAA: " << CONF_AAA << "\n";
    strSimInfo << "CONF_DIFF_FLUC_COEF: " << CONF_DIFF_FLUC_COEF << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif// FENE_P
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

    if(step == REPORT_SAVE){ //check if first time step to save data
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

/*
__host__
void loadMoments(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST
    dfloat* C
    #endif 
){
    size_t indexMacr;


    //first moments
    dfloat rhoVar, uxVar, uyVar, uzVar;
    dfloat pixx, pixy, pixz, piyy, piyz, pizz;
    dfloat invRho;
    dfloat pop[Q];
    #ifdef OMEGA_FIELD
    dfloat omegaVar;
    #endif
    #ifdef SECOND_DIST 
    dfloat cVar, invC, qx_t30, qy_t30, qz_t30;
    dfloat gNode[GQ];
    #endif

    


    for(int z = 0; z< NZ;z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                indexMacr = idxScalarGlobal(x,y,z);

                rhoVar = rho[indexMacr];
                h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = rhoVar-RHO_0;
                uxVar = ux[indexMacr];
                h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = F_M_I_SCALE*uxVar;
                uyVar = uy[indexMacr];
                h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = F_M_I_SCALE*uyVar;
                uzVar = uz[indexMacr];
                h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = F_M_I_SCALE*uzVar;


                //second moments
                //define equilibrium populations
                for (int i = 0; i < Q; i++)
                {
                    pop[i] = gpu_f_eq(w[i] * RHO_0,
                                    3 * (uxVar * cx[i] + uyVar * cy[i] + uzVar * cz[i]),
                                    1 - 1.5 * (uxVar * uxVar + uyVar * uyVar + uzVar * uzVar));
                }


                invRho = 1.0/rhoVar;
                pixx =  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
                pixy = ((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho;
                pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
                piyy =  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
                piyz = ((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho;
                pizz =  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*pixx;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*pixy;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*pixz;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*piyy;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_IJ_SCALE*piyz;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = F_M_II_SCALE*pizz;


                #ifdef OMEGA_FIELD
                omegaVar = omega[indexMacr];
                h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = omegaVar; 
                #endif

                #ifdef SECOND_DIST 
                cVar = C[indexMacr];
                h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = cVar;

                dfloat udx_t30 = G_DIFF_FLUC_COEF * (qx_t30*invC - uxVar*F_M_I_SCALE);
                dfloat udy_t30 = G_DIFF_FLUC_COEF * (qy_t30*invC - uyVar*F_M_I_SCALE);
                dfloat udz_t30 = G_DIFF_FLUC_COEF * (qz_t30*invC - uzVar*F_M_I_SCALE);

                dfloat multiplyTerm = cVar * gW0;
                dfloat pics2 = 1.0;

                gNode[ 0] = multiplyTerm * (pics2);
                multiplyTerm = cVar * gW1;
                gNode[ 1] = multiplyTerm * (pics2 + uxVar * F_M_I_SCALE  + udx_t30 );
                gNode[ 2] = multiplyTerm * (pics2 - uxVar * F_M_I_SCALE  - udx_t30 );
                gNode[ 3] = multiplyTerm * (pics2 + uyVar * F_M_I_SCALE  + udy_t30 );
                gNode[ 4] = multiplyTerm * (pics2 - uyVar * F_M_I_SCALE  - udy_t30 );
                gNode[ 5] = multiplyTerm * (pics2 + uzVar * F_M_I_SCALE  + udz_t30 );
                gNode[ 6] = multiplyTerm * (pics2 - uzVar * F_M_I_SCALE  - udz_t30 );
                multiplyTerm = cVar * gW2;
                gNode[ 7] = multiplyTerm * (pics2 + uxVar * F_M_I_SCALE + uyVar * F_M_I_SCALE + udx_t30 + udy_t30 );
                gNode[ 8] = multiplyTerm * (pics2 - uxVar * F_M_I_SCALE - uyVar * F_M_I_SCALE - udx_t30 - udy_t30 );
                gNode[ 9] = multiplyTerm * (pics2 + uxVar * F_M_I_SCALE + uzVar * F_M_I_SCALE + udx_t30 + udz_t30 );
                gNode[10] = multiplyTerm * (pics2 - uxVar * F_M_I_SCALE - uzVar * F_M_I_SCALE - udx_t30 - udz_t30 );
                gNode[11] = multiplyTerm * (pics2 + uyVar * F_M_I_SCALE + uzVar * F_M_I_SCALE + udy_t30 + udz_t30 );
                gNode[12] = multiplyTerm * (pics2 - uyVar * F_M_I_SCALE - uzVar * F_M_I_SCALE - udy_t30 - udz_t30 );
                gNode[13] = multiplyTerm * (pics2 + uxVar * F_M_I_SCALE - uyVar * F_M_I_SCALE + udx_t30 - udy_t30 );
                gNode[14] = multiplyTerm * (pics2 - uxVar * F_M_I_SCALE + uyVar * F_M_I_SCALE - udx_t30 + udy_t30 );
                gNode[15] = multiplyTerm * (pics2 + uxVar * F_M_I_SCALE - uzVar * F_M_I_SCALE + udx_t30 - udz_t30 );
                gNode[16] = multiplyTerm * (pics2 - uxVar * F_M_I_SCALE + uzVar * F_M_I_SCALE - udx_t30 + udz_t30 );
                gNode[17] = multiplyTerm * (pics2 + uyVar * F_M_I_SCALE - uzVar * F_M_I_SCALE + udy_t30 - udz_t30 );
                gNode[18] = multiplyTerm * (pics2 - uyVar * F_M_I_SCALE + uzVar * F_M_I_SCALE - udy_t30 + udz_t30 );

                qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
                qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
                qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));




                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qx_t30;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qy_t30;
                h_fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = qz_t30;
                #endif



            }
        }
    }
}


__host__
void loadSimField(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST
    dfloat* C
    #endif 
){
    std::string strFileRho, strFileUx, strFileUy, strFileUz;
    std::string strFileOmega;
    std::string strFileC;
    std::string strFileBc; 
    std::string strFileFx, strFileFy, strFileFz;

    strFileRho = getVarFilename("rho", LOAD_FIELD_STEP, ".bin");
    strFileUx = getVarFilename("ux", LOAD_FIELD_STEP, ".bin");
    strFileUy = getVarFilename("uy", LOAD_FIELD_STEP, ".bin");
    strFileUz = getVarFilename("uz", LOAD_FIELD_STEP, ".bin");
    #ifdef OMEGA_FIELD
    strFileOmega = getVarFilename("omega", LOAD_FIELD_STEP, ".bin");
    #endif
    #ifdef SECOND_DIST 
    strFileC = getVarFilename("C", LOAD_FIELD_STEP, ".bin");
    #endif
    #if NODE_TYPE_SAVE
    strFileBc = getVarFilename("bc", LOAD_FIELD_STEP, ".bin");
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    strFileFx = getVarFilename("fx", LOAD_FIELD_STEP, ".bin");
    strFileFy = getVarFilename("fy", LOAD_FIELD_STEP, ".bin");
    strFileFz = getVarFilename("fz", LOAD_FIELD_STEP, ".bin");
    #endif

    // load files
    loadVarBin(strFileRho, rho, MEM_SIZE_SCALAR, false);
    loadVarBin(strFileUx, ux, MEM_SIZE_SCALAR, false);
    loadVarBin(strFileUy, uy, MEM_SIZE_SCALAR, false);
    loadVarBin(strFileUz, uz, MEM_SIZE_SCALAR, false);
    #ifdef OMEGA_FIELD
    loadVarBin(strFileOmega, omega, MEM_SIZE_SCALAR, false);
    #endif
    #ifdef SECOND_DIST
    loadVarBin(strFileC, C, MEM_SIZE_SCALAR, false);
    #endif
    #if NODE_TYPE_SAVE
    loadVarBin(strFileBc, nodeTypeSave, MEM_SIZE_SCALAR, false);
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    loadVarBin(strFileFx, h_BC_Fx, MEM_SIZE_SCALAR, false);
    loadVarBin(strFileFy, h_BC_Fy, MEM_SIZE_SCALAR, false);
    loadVarBin(strFileFz, h_BC_Fz, MEM_SIZE_SCALAR, false);
    #endif


    loadMoments(h_fMom,rho,ux,uy,uz, OMEGA_FIELD_PARAMS_DECLARATION
            #ifdef SECOND_DIST
            C
            #endif 
            );

}


void loadVarBin(
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
        fread(var, memSize, 1, outFile);
        fclose(outFile);
    }
    else
    {
        printf("Error loading \"%s\" \nProbably wrong path!\n", strFile.c_str());
    }
}
*/