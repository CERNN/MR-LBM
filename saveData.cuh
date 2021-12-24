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

#include "globalFunctions.h"

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
void linearMacr(
    dfloat* h_fMom,
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
    unsigned int step
);

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
void saveMacr(
    dfloat* rho,
    dfloat* ux,
    dfloat* uy,
    dfloat* uz,
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
std::string getSimInfoString(int step);

/*
*   Save simulation's information
*   @param info: simulation's informations
*/
void saveSimInfo(int step);

#endif __SAVE_DATA_H