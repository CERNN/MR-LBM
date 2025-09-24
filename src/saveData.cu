#include "saveData.cuh"
#ifdef OMEGA_FIELD
#include "nnf.h"
#endif //OMEGA_FIELD

std::filesystem::path getExecutablePath() {
    #if defined(_WIN32)
        char result[MAX_PATH];
        DWORD count = GetModuleFileNameA(NULL, result, MAX_PATH);
        if (count == 0) throw std::runtime_error("Error obtaining path to executable (Windows).");
        return std::filesystem::path(std::string(result, count));
    #elif defined(__linux__)
        char result[1024];
        ssize_t count = readlink("/proc/self/exe", result, sizeof(result));
        if (count == -1) throw std::runtime_error("Error obtaining path to executable (Linux).");
        return std::filesystem::path(std::string(result, count));
    #elif defined(__APPLE__)
        char result[1024];
        uint32_t size = sizeof(result);
        if (_NSGetExecutablePath(result, &size) != 0)
            throw std::runtime_error("Error obtaining path to executable  (macOS).");
        return std::filesystem::path(result);
    #else
        #error "Platform not supported"
    #endif
}

std::filesystem::path folderSetup()
{
    std::filesystem::path exePath = getExecutablePath();
    std::filesystem::path binDir = exePath.parent_path();

    std::filesystem::path baseDir = binDir / PATH_FILES / ID_SIM;
    std::filesystem::create_directories(baseDir);

    return baseDir;
}   

// choose correct swap based on sizeof(dfloat)
template<typename T>
void writeBigEndian(std::ofstream& ofs, const T* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if constexpr (sizeof(T) == 4) {
            uint32_t tmp;
            memcpy(&tmp, &data[i], 4);
            tmp = swap32(tmp);
            ofs.write(reinterpret_cast<char*>(&tmp), 4);
        }
        else if constexpr (sizeof(T) == 8) {
            uint64_t tmp;
            memcpy(&tmp, &data[i], 8);
            tmp = swap64(tmp);
            ofs.write(reinterpret_cast<char*>(&tmp), 8);
        }
    }
}

__host__
void saveMacr(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    unsigned int* hNodeType,
    OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif //SECOND_DIST
    #ifdef A_XX_DIST 
    dfloat* Axx,
    #endif //A_XX_DIST
    #ifdef A_XY_DIST 
    dfloat* Axy,
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST 
    dfloat* Axz,
    #endif //A_XZ_DIST
    #ifdef A_YY_DIST 
    dfloat* Ayy,
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST 
    dfloat* Ayz,
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST 
    dfloat* Azz,
    #endif //A_ZZ_DIST
    #ifdef LOG_CONFORMATION
        #ifdef A_XX_DIST
        dfloat* Cxx,
        #endif //A_XX_DIST
        #ifdef A_XY_DIST
        dfloat* Cxy,
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST
        dfloat* Cxz,
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST
        dfloat* Cyy,
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST
        dfloat* Cyz,
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST
        dfloat* Czz,
        #endif //A_ZZ_DIST
    #endif //LOG_CONFORMATION
    NODE_TYPE_SAVE_PARAMS_DECLARATION
    BC_FORCES_PARAMS_DECLARATION(h_) 
    unsigned int nSteps
){


    //linearize
    size_t indexMacr;
    for(int z = 0; z< NZ;z++){
        for(int y = 0; y< NY;y++){
            for(int x = 0; x< NX;x++){
                indexMacr = idxScalarGlobal(x,y,z);

                rho[indexMacr] = RHO_0+h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_RHO_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                ux[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UX_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UY_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                uz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_UZ_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];

                #ifdef OMEGA_FIELD
                omega[indexMacr] = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M_OMEGA_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]; 
                #endif //OMEGA_FIELD

                #ifdef SECOND_DIST 
                C[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, M2_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)];
                #endif //SECOND_DIST
                #ifdef A_XX_DIST 
                Axx[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XX_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif //A_XX_DIST
                #ifdef A_XY_DIST 
                Axy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif //A_XY_DIST
                #ifdef A_XZ_DIST 
                Axz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_XZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif //A_XZ_DIST
                #ifdef A_YY_DIST 
                Ayy[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YY_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif //A_YY_DIST
                #ifdef A_YZ_DIST 
                Ayz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_YZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif //A_YZ_DIST
                #ifdef A_ZZ_DIST 
                Azz[indexMacr]  = h_fMom[idxMom(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, A_ZZ_C_INDEX, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] - CONF_ZERO;
                #endif //A_ZZ_DIST
                
                #if NODE_TYPE_SAVE
                nodeTypeSave[indexMacr] = (dfloat)hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)]; 
                #endif //NODE_TYPE_SAVE

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
    #endif // BC_FORCES && SAVE_BC_FORCES


    // Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz; 
    std::string strFileOmega;
    std::string strFileC;
    std::string strFileBc; 
    std::string strFileFx, strFileFy, strFileFz;
    std::string strFileAxx, strFileAxy, strFileAxz, strFileAyy, strFileAyz, strFileAzz;


    if (VTK_SAVE){
        std::string strFileVtk, strFileVtr;
        strFileVtk = getVarFilename("vtk", nSteps, ".vtk");

        saveVarVTK(
                strFileVtk, 
                rho,ux,uy,uz, OMEGA_FIELD_PARAMS
                    #ifdef SECOND_DIST 
                    C,
                    #endif //SECOND_DIST
                    #ifdef A_XX_DIST 
                    Axx,
                    #endif //A_XX_DIST
                    #ifdef A_XY_DIST 
                    Axy,
                    #endif //A_XY_DIST
                    #ifdef A_XZ_DIST 
                    Axz,
                    #endif //A_XZ_DIST
                    #ifdef A_YY_DIST 
                    Ayy,
                    #endif //A_YY_DIST
                    #ifdef A_YZ_DIST 
                    Ayz,
                    #endif //A_YZ_DIST
                    #ifdef A_ZZ_DIST 
                    Azz,
                    #endif //A_ZZ_DIST
                    NODE_TYPE_SAVE_PARAMS BC_FORCES_PARAMS(h_) 
                    nSteps     
                );
    }
    if (BIN_SAVE){
        strFileRho = getVarFilename("rho", nSteps, ".bin");
        strFileUx = getVarFilename("ux", nSteps, ".bin");
        strFileUy = getVarFilename("uy", nSteps, ".bin");
        strFileUz = getVarFilename("uz", nSteps, ".bin");

        #ifdef OMEGA_FIELD
        strFileOmega = getVarFilename("omega", nSteps, ".bin");
        #endif //OMEGA_FIELD
        #ifdef SECOND_DIST 
        strFileC = getVarFilename("C", nSteps, ".bin");
        #endif //SECOND_DIST
        #ifdef A_XX_DIST 
        strFileAxx = getVarFilename("Axx", nSteps, ".bin");
        #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        strFileAxy = getVarFilename("Axy", nSteps, ".bin");
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST 
        strFileAxz = getVarFilename("Axz", nSteps, ".bin");
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST 
        strFileAyy = getVarFilename("Ayy", nSteps, ".bin");
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        strFileAyz = getVarFilename("Ayz", nSteps, ".bin");
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        strFileAzz = getVarFilename("Azz", nSteps, ".bin");
        #endif //A_ZZ_DIST
        #if NODE_TYPE_SAVE
        strFileBc = getVarFilename("bc", nSteps, ".bin");
        #endif //NODE_TYPE_SAVE
        #if defined BC_FORCES && defined SAVE_BC_FORCES
        strFileFx = getVarFilename("fx", nSteps, ".bin");
        strFileFy = getVarFilename("fy", nSteps, ".bin");
        strFileFz = getVarFilename("fz", nSteps, ".bin");
        #endif //BC_FORCES &&  SAVE_BC_FORCES
        // saving files
        saveVarBin(strFileRho, rho, MEM_SIZE_SCALAR, false);
        saveVarBin(strFileUx, ux, MEM_SIZE_SCALAR, false);
        saveVarBin(strFileUy, uy, MEM_SIZE_SCALAR, false);
        saveVarBin(strFileUz, uz, MEM_SIZE_SCALAR, false);
        #ifdef OMEGA_FIELD
        saveVarBin(strFileOmega, omega, MEM_SIZE_SCALAR, false);
        #endif //OMEGA_FIELD
        #ifdef SECOND_DIST
        saveVarBin(strFileC, C, MEM_SIZE_SCALAR, false);
        #endif //SECOND_DIST
        #ifdef A_XX_DIST 
        saveVarBin(strFileAxx, Axx, MEM_SIZE_SCALAR, false);
        #endif //A_XX_DIST
        #ifdef A_XY_DIST 
        saveVarBin(strFileAxy, Axy, MEM_SIZE_SCALAR, false);
        #endif //A_XY_DIST
        #ifdef A_XZ_DIST 
        saveVarBin(strFileAxz, Axz, MEM_SIZE_SCALAR, false);
        #endif //A_XZ_DIST
        #ifdef A_YY_DIST 
        saveVarBin(strFileAyy, Ayy, MEM_SIZE_SCALAR, false);
        #endif //A_YY_DIST
        #ifdef A_YZ_DIST 
        saveVarBin(strFileAyz, Ayz, MEM_SIZE_SCALAR, false);
        #endif //A_YZ_DIST
        #ifdef A_ZZ_DIST 
        saveVarBin(strFileAzz, Azz, MEM_SIZE_SCALAR, false);
        #endif //A_ZZ_DIST
        
        #if NODE_TYPE_SAVE
        saveVarBin(strFileBc, (dfloat*)nodeTypeSave, MEM_SIZE_SCALAR, false);
        #endif //NODE_TYPE_SAVE
        #if defined BC_FORCES && defined SAVE_BC_FORCES
        saveVarBin(strFileFx, h_BC_Fx, MEM_SIZE_SCALAR, false);
        saveVarBin(strFileFy, h_BC_Fy, MEM_SIZE_SCALAR, false);
        saveVarBin(strFileFz, h_BC_Fz, MEM_SIZE_SCALAR, false);
        #endif //BC_FORCES && SAVE_BC_FORCES
    }
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


std::vector<float> convertPointToCellScalar(
    const float* pointField, size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<float> cellField(Ncells, 0.0f);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        float sum=0.0f;
        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++)
            sum += pointField[idxScalarGlobal(x+dx, y+dy, z+dz)];
        cellField[cidx] = sum/8.0f;
    }
    return cellField;
}

std::vector<dfloat3> convertPointToCellVector(
    const float* ux, const float* uy, const float* uz,
    size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<dfloat3> cellField(Ncells);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        float sumx=0.0f, sumy=0.0f, sumz=0.0f;
        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++) {
            size_t pidx = idxScalarGlobal(x+dx, y+dy, z+dz);
            sumx += ux[pidx]; sumy += uy[pidx]; sumz += uz[pidx];
        }
        cellField[cidx] = { sumx/8.0f, sumy/8.0f, sumz/8.0f };
    }
    return cellField;
}

std::vector<dfloat6> convertPointToCellTensor6(
    const float* Axx, const float* Ayy, const float* Azz,
    const float* Axy, const float* Ayz, const float* Axz,
    size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<dfloat6> cellField(Ncells);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        float sumxx=0,sumyy=0,sumzz=0,sumxy=0,sumyz=0,sumxz=0;
        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++) {
            size_t pidx = idxScalarGlobal(x+dx, y+dy, z+dz);
            sumxx += Axx[pidx]; sumyy += Ayy[pidx]; sumzz += Azz[pidx];
            sumxy += Axy[pidx]; sumyz += Ayz[pidx]; sumxz += Axz[pidx];
        }
        cellField[cidx] = { sumxx/8.0f, sumyy/8.0f, sumzz/8.0f,
                            sumxy/8.0f, sumyz/8.0f, sumxz/8.0f };
    }
    return cellField;
}

std::vector<int> convertPointToCellIntMode(
    const int* pointField, size_t NX, size_t NY, size_t NZ)
{
    size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
    std::vector<int> cellField(Ncells, 0);

    for (size_t z=0; z<NZ-1; z++)
    for (size_t y=0; y<NY-1; y++)
    for (size_t x=0; x<NX-1; x++) {
        size_t cidx = x + y*(NX-1) + z*(NX-1)*(NY-1);
        std::map<int,int> counts;

        for(int dz=0; dz<=1; dz++)
        for(int dy=0; dy<=1; dy++)
        for(int dx=0; dx<=1; dx++)
            counts[pointField[idxScalarGlobal(x+dx, y+dy, z+dz)]]++;

        int mode=0,maxCount=0;
        for(auto &kv : counts)
            if(kv.second>maxCount) { maxCount=kv.second; mode=kv.first; }

        cellField[cidx] = mode;
    }
    return cellField;
}

void saveVarVTK(
    std::string filename, 
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif //SECOND_DIST
    #ifdef A_XX_DIST 
    dfloat* Axx,
    #endif //A_XX_DIST
    #ifdef A_XY_DIST 
    dfloat* Axy,
    #endif //A_XY_DIST
    #ifdef A_XZ_DIST 
    dfloat* Axz,
    #endif //A_XY_DIST
    #ifdef A_YY_DIST 
    dfloat* Ayy,
    #endif //A_YY_DIST
    #ifdef A_YZ_DIST 
    dfloat* Ayz,
    #endif //A_YZ_DIST
    #ifdef A_ZZ_DIST 
    dfloat* Azz,
    #endif //A_ZZ_DIST
    NODE_TYPE_SAVE_PARAMS_DECLARATION
    BC_FORCES_PARAMS_DECLARATION(h_) 
    unsigned int nSteps 
    )
{

    if(!CELLDATA_SAVE){
        //printf("Saving VTK in POINT_DATA format");
        const size_t N = NX*NY*NZ;
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) throw std::runtime_error("Cannot open " + filename);

        //Header 
        ofs << "# vtk DataFile Version 3.0\n"
            << "LBM output (binary)\n"
            << "BINARY\n"                               // ← here!
            << "DATASET STRUCTURED_POINTS\n"
            << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n"
            << "ORIGIN 0 0 0\n"
            << "SPACING 1 1 1\n"
            << "POINT_DATA " << N << "\n";
        ofs << "SCALARS rho float 1\n"
            << "LOOKUP_TABLE default\n";
        writeBigEndian(ofs, rho, N);

        ofs << "VECTORS velocity float\n";
        for (size_t i = 0; i < N; ++i) {
            dfloat v[3] = { ux[i]/F_M_I_SCALE, uy[i]/F_M_I_SCALE, uz[i]/F_M_I_SCALE};
            writeBigEndian(ofs, v, 3);
        }

        #ifdef OMEGA_FIELD
            ofs << "SCALARS omega float 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, omega, N);
        #endif //OMEGA_FIELD

        #ifdef SECOND_DIST
            ofs << "SCALARS C float 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, C, N);
        #endif //SECOND_DIST

        #ifdef CONFORMATION_TENSOR
            ofs << "TENSORS6 Aij float\n";
            for (size_t i = 0; i < N; ++i) {
                dfloat tensor[6] = {
                    Axx[i], Ayy[i], Azz[i],
                    Axy[i], Ayz[i], Axz[i]
                };
                writeBigEndian(ofs, tensor, 6);
            }
        #endif //CONFORMATION_TENSOR

        #ifdef SAVE_BC_FORCES
            ofs << "VECTORS forces float\n";
            for (size_t i = 0; i < N; ++i) {
                dfloat f[3] = { fx[i], fy[i], fz[i] };
                writeBigEndian(ofs, f, 3);
            }
        #endif //SAVE_BC_FORCES

        #if NODE_TYPE_SAVE
            ofs << "SCALARS bc int 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, NODE_TYPE_SAVE_PARAMS N);
        #endif //NODE_TYPE_SAVE
    }else{ 
        //printf("Saving VTK in CELL_DATA format");
        const size_t Ncells = (NX-1)*(NY-1)*(NZ-1);
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) throw std::runtime_error("Cannot open " + filename);

        //Header 
        ofs << "# vtk DataFile Version 3.0\n"
            << "LBM output (binary)\n"
            << "BINARY\n"                               // ← here!
            << "DATASET STRUCTURED_POINTS\n"
            << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\n"
            << "ORIGIN 0 0 0\n"
            << "SPACING 1 1 1\n"
            << "CELL_DATA " << Ncells << "\n";
        auto rho_cell = convertPointToCellScalar(rho,NX,NY,NZ);
        ofs << "SCALARS rho float 1\n"
            << "LOOKUP_TABLE default\n";
        writeBigEndian(ofs, rho_cell.data(), rho_cell.size());

        auto vel_cell = convertPointToCellVector(ux,uy,uz,NX,NY,NZ);
        ofs << "VECTORS velocity float\n";
        for(size_t i=0;i<Ncells;i++){
            float v[3] = { vel_cell[i].x/F_M_I_SCALE,
                        vel_cell[i].y/F_M_I_SCALE,
                        vel_cell[i].z/F_M_I_SCALE };
            writeBigEndian(ofs,v,3);
        }

        #ifdef OMEGA_FIELD
            auto omega_cell = convertPointToCellScalar(omega,NX,NY,NZ);
            ofs << "SCALARS omega float 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, omega_cell.data(), omega_cell.size());
        #endif //OMEGA_FIELD

        #ifdef SECOND_DIST
            auto C_cell = convertPointToCellScalar(C,NX,NY,NZ);
            ofs << "SCALARS C float 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, C_cell.data(), Ncells);
        #endif //SECOND_DIST

        #ifdef CONFORMATION_TENSOR
            auto A_cell = convertPointToCellTensor6(Axx,Ayy,Azz,Axy,Ayz,Axz,NX,NY,NZ);
            ofs << "TENSORS6 Aij float\n";
            for (size_t i = 0; i < Ncells; ++i) {
                dfloat tensor[6] = {
                    A_cell[i].xx,A_cell[i].yy,A_cell[i].zz,
                    A_cell[i].xy,A_cell[i].yz,A_cell[i].xz
                };
                writeBigEndian(ofs, tensor, 6);
            }
        #endif //CONFORMATION_TENSOR

        #ifdef SAVE_BC_FORCES
            auto f_cell = convertPointToCellVector(fx, fy, fz,NX,NY,NZ);
            ofs << "VECTORS forces float\n";
            for (size_t i = 0; i < Ncells; ++i) {
                dfloat f[3] = { fx[i], fy[i], fz[i] };
                writeBigEndian(ofs, f, 3);
            }
        #endif //SAVE_BC_FORCES

        #if NODE_TYPE_SAVE
            auto bc_cell = convertPointToCellIntMode(NODE_TYPE_SAVE,NX,NY,NZ);
            ofs << "SCALARS bc int 1\n"
                << "LOOKUP_TABLE default\n";
            writeBigEndian(ofs, bc_cell.data(), Ncells);
        #endif //NODE_TYPE_SAVE
    }  
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
    
    std::filesystem::path baseDir = folderSetup();

    std::string baseName = ID_SIM + std::string("_") + varName;

    std::string strFile = (baseDir / baseName).string();

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
    #endif //SINGLE_PRECISION
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
    #endif //BC_MOMENT_BASED
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
    std::filesystem::path baseDir = folderSetup();

    std::string baseName = ID_SIM + std::string("_info.txt");
    std::filesystem::path strInf =  (baseDir / baseName).string();

    FILE* outFile = nullptr;

    outFile = fopen(strInf.string().c_str(), "w");
    if(outFile != nullptr)
    {
        std::string strSimInfo = getSimInfoString(step,MLUPS);
        fprintf(outFile, strSimInfo.c_str());
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strInf.string().c_str());
    }
    
}
/**/


void saveTreatData(std::string fileName, std::string dataString, int step)
{
    #if SAVEDATA
    std::filesystem::path baseDir = folderSetup();;

    std::filesystem::path strInf = baseDir / (ID_SIM + fileName + ".txt");

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
    #endif //SAVEDATA
    #if CONSOLEPRINT
    printf("%s \n",dataString.c_str());
    #endif //CONSOLEPRINT
}
