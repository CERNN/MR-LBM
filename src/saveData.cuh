#ifndef __SAVE_DATA_H
#define __SAVE_DATA_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <vector>

#include "globalFunctions.h"
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "errorDef.h"
#include "globalStructs.h"


/*
*   @brief Change field vector order to be used saved in binary
*   @param h_fMom: host macroscopic field based on block and thread index
*   @param rho: rho field
*   @param ux: ux field
*   @param uy: uy field
*   @param uz: uz field
*   @param nSteps: number of steps of the simulation
*/
__host__
void linearMacr(dfloat* h_fMom, dfloat* rho, dfloat* ux, dfloat* uy, dfloat* uz, OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST 
    dfloat* C,
    #endif 
    #if NODE_TYPE_SAVE
    dfloat* nodeTypeSave,
    unsigned int* hNodeType,
    #endif
    #if defined BC_FORCES && defined SAVE_BC_FORCES
    dfloat* h_BC_Fx,
    dfloat* h_BC_Fy,
    dfloat* h_BC_Fz,
    #endif
    unsigned int step
);

/*
__host__
void loadMoments(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz, OMEGA_FIELD_PARAMS_DECLARATION
    #ifdef SECOND_DIST
    dfloat* C
    #endif 
);

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
);

void loadVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append);
*/

/*
*   @brief Save all macroscopics in binary format
*   @param rho: rho field
*   @param ux: ux field
*   @param uy: uy field
*   @param uz: uz field
*   @param nSteps: number of steps of the simulation
*   @obs Check CPU endianess
*   @obs The initial position of the array is x=0 and y=0 and z=0, 
*        so the variables starts on SWF and ends in NEB
*/
__host__
void saveMacr(dfloat* h_fMom, dfloat* rho, dfloat* ux, dfloat* uy, dfloat* uz, unsigned int* hNodeType, OMEGA_FIELD_PARAMS_DECLARATION
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
);

/*
*   @brief Save array content to binary file
*   @param strFile: filename to save
*   @param var: float variable to save
*   @param memSize: sizeof var
*   @param append: content must be appended to file or overwrite
*/
void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append
);

template<typename T>
void writeBigEndian(std::ofstream& ofs, const T* data, size_t count);

/*
*   @brief Save field on vtk file
*   @param var_name: name of the variable
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
    NODE_TYPE_SAVE_PARAMS_DECLARATION
    BC_FORCES_PARAMS_DECLARATION(h_) 
    unsigned int nSteps
);

/*
*   @brief Get variable filename
*   @param var_name: name of the variable
*   @param step: steps number of the file
*   @param ext: file extension (with dot, e.g. ".bin", ".csv")
*   @return filename string
*/
std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext
);

/*
*   @brief Setup folder to save variables
*/
void folderSetup();

/*
*   Get string with simulation information
*   @param step: simulation's step
*   @return string with simulation info
*/
std::string getSimInfoString(int step,dfloat MLUPS);

/*
*   Save simulation's information
*   @param info: simulation's informations
*/
void saveSimInfo(int step,dfloat MLUPS);



/*
*   @brief Saves in a ".txt" file the required treated data
*   @param fileName: primary file name
*   @param dataString: dataString to be saved
*   @param step: time step
*/
void saveTreatData(std::string fileName, std::string dataString, int step);


#endif __SAVE_DATA_H