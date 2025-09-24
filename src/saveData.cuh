/**
 *  @file saveData.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Save data
 *  @version 0.4.0
 *  @date 01/09/2025
 */


#ifndef __SAVE_DATA_H
#define __SAVE_DATA_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

#include <map>
#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <vector>
#include <filesystem>
#if defined(_WIN32)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#endif


#include "globalFunctions.h"
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "errorDef.h"
#include "globalStructs.h"

__host__
std::filesystem::path getExecutablePath();

__host__
std::filesystem::path folderSetup();

template<typename T>
void writeBigEndian(std::ofstream& ofs, const T* data, size_t count);


/**
 *  @brief Change field vector order to be used saved in binary
 *  @param h_fMom: Pointer to the host array containing the current macroscopic moments.
 *  @param rho: rho field
 *  @param ux: ux field
 *  @param uy: uy field
 *  @param uz: uz field
 *  @param step: Current time step
*/
__host__
void linearMacr(dfloat* h_fMom, dfloat* rho, dfloat* ux, dfloat* uy, dfloat* uz, OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif //SECOND_DIST
    #if NODE_TYPE_SAVE
    dfloat* nodeTypeSave,
    unsigned int* hNodeType,
    #endif //NODE_TYPE_SAVE
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    dfloat* h_BC_Fx,
    dfloat* h_BC_Fy,
    dfloat* h_BC_Fz,
    #endif // BC_FORCES && SAVE_BC_FORCES
    unsigned int step
);


/**
 *  @brief Save all macroscopics in binary format
 *  @param rho: rho field
 *  @param ux: ux field
 *  @param uy: uy field
 *  @param uz: uz field
 *  @param hNodeType: Pointer to the host array containing the current node types.
 *  @param nSteps: number of steps of the simulation
*/
__host__
void saveMacr(dfloat* h_fMom, dfloat* rho, dfloat* ux, dfloat* uy, dfloat* uz, unsigned int* hNodeType, OMEGA_FIELD_PARAMS_DECLARATION
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
    NODE_TYPE_SAVE_PARAMS_DECLARATION
    BC_FORCES_PARAMS_DECLARATION(h_) 
    unsigned int nSteps
);

/*
 *  @brief Save array content to binary file
 *  @param strFile: filename to save
 *  @param var: float variable to save
 *  @param memSize: sizeof var
 *  @param append: content must be appended to file or overwrite
*/
void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append
);


/**
 *  @brief Convert point-based scalar data into cell-centered scalar data
 *  @param pointField: pointer to array of scalar values defined at grid points
 *  @param NX: number of grid points in X direction
 *  @param NY: number of grid points in Y direction
 *  @param NZ: number of grid points in Z direction
 *  @return Vector with scalar values averaged/interpolated at cell centers
*/
std::vector<float> convertPointToCellScalar(
    const float* pointField,
    size_t NX,
    size_t NY,
    size_t NZ
);

/**
 *  @brief Convert point-based vector field into cell-centered vector field
 *  @param ux: pointer to array with X-component of velocity/field at grid points
 *  @param uy: pointer to array with Y-component of velocity/field at grid points
 *  @param uz: pointer to array with Z-component of velocity/field at grid points
 *  @param NX: number of grid points in X direction
 *  @param NY: number of grid points in Y direction
 *  @param NZ: number of grid points in Z direction
 *  @return Vector of dfloat3 objects with interpolated values at cell centers
*/
std::vector<dfloat3> convertPointToCellVector(
    const float* ux,
    const float* uy,
    const float* uz,
    size_t NX,
    size_t NY,
    size_t NZ
);

/**
 *  @brief Converts point-based tensors (point grid) to cell-based tensors
 *  @param Axx Pointer to the array containing the xx component values at points
 *  @param Ayy Pointer to the array containing the yy component values at points
 *  @param Azz Pointer to the array containing the zz component values at points
 *  @param Axy Pointer to the array containing the xy component values at points
 *  @param Ayz Pointer to the array containing the yz component values at points
 *  @param Axz Pointer to the array containing the xz component values at points
 *  @param NX: number of grid points in X direction
 *  @param NY: number of grid points in Y direction
 *  @param NZ: number of grid points in Z direction
 *  @return std::vector<dfloat6> Vector containing the averaged tensors of each cell
 */
std::vector<dfloat6> convertPointToCellTensor6(
    const float* Axx, 
    const float* Ayy,
    const float* Azz,
    const float* Axy,
    const float* Ayz, 
    const float* Axz,
    size_t NX, 
    size_t NY, 
    size_t NZ
);

/**
 *  @brief Converts point-based integer data into cell-centered integer data
 *  @param pointField Pointer to the array of integer values defined at grid points
 *  @param NX Number of points in the X direction
 *  @param NY Number of points in the Y direction
 *  @param NZ Number of points in the Z direction
 *  @return std::vector<int> Vector containing integer values aggregated at cell centers
 */
std::vector<int> convertPointToCellIntMode(
    const int* pointField,
    size_t NX,
    size_t NY,
    size_t NZ
);

/*
 *  @brief Save field on vtk file
 *  @param var_name: name of the variable
*/
void saveVarVTK(
    std::string strFileVtk, 
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
    NODE_TYPE_SAVE_PARAMS_DECLARATION
    BC_FORCES_PARAMS_DECLARATION(h_) 
    unsigned int nSteps
);

/**
 *  @brief Get variable filename
 *  @param var_name: name of the variable
 *  @param step: steps number of the file
 *  @param ext: file extension (with dot, e.g. ".bin", ".csv")
 *  @return filename string
*/
std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext
);

/**
*   Get string with simulation information
 *  @param step: simulation's step
 *  @param MLUPS: Mega Lattice Updates Per Second
 *  @return string with simulation info
*/
std::string getSimInfoString(int step,dfloat MLUPS);

/**
*   Save simulation's information
 *  @param info: simulation's informations
 *  @param MLUPS: Mega Lattice Updates Per Second
*/
void saveSimInfo(int step,dfloat MLUPS);



/**
 *  @brief Saves in a ".txt" file the required treated data
 *  @param fileName: primary file name
 *  @param dataString: dataString to be saved
 *  @param step: time step
 *  @param headerExist: if the header already exists in the file (default is false)
*/
void saveTreatData(std::string fileName, std::string dataString, int step, bool headerExist = false);

/**
 * @brief Saves in a ".txt" file the header of the treated data
 * @param fileName: primary file name
 * @param headerString: header string to be saved
 */
void saveTreatDataHeader(std::string fileName, std::string headerString);


#endif //__SAVE_DATA_H