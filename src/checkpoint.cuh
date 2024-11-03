/*
*   @file checkpoint.cuh
*   @author Marco A Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Loading and saving simulation checkpoints
*   @version 0.1.0
*   @date 16/02/2023
*/

#ifndef __CHECKPOINT_H
#define __CHECKPOINT_H

#include <string>
#include <fstream>
#include <iostream>     // std::cout, std::fixed
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "errorDef.h"
#include "globalFunctions.h"
#include "var.h"



#define __LOAD_CHECKPOINT 1
#define __SAVE_CHECKPOINT 2

/**
*   @brief Create a new folder
*   
*   @param foldername Folder name
*   @return size_t Filesize
*/
void createFolder(
    std::string foldername
);

/**
*   @brief Get the filesize of file
*   
*   @param filename Filename to get size
*   @return size_t Filesize
*/
size_t getFileSize(
    std::string filename
);

/**
*   @brief Reads file content into GPU array
*
*   @param arr GPU array to write file content to
*   @param filename Filename to read from
*   @param arr_size_bytes Size in bytes to read from file. If zero, reads whole file
*   @param tmp Temporary array used to read file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__
void readFileIntoArray(
    void* arr, 
    std::string filename,
    size_t arr_size_bytes, 
    void* tmp
);

/**
*   @brief Writes GPU array content into file
*
*   @param arr GPU array to read content from
*   @param filename Filename to write to
*   @param arr_size_bytes Size in bytes to write to file
*   @param tmp Temporary array used to write to file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__
void writeFileIntoArray(
    void* arr, 
    const std::string filename, 
    const size_t arr_size_bytes, 
    void* tmp
);

/**
*   @brief Get the checkpoint filename to read from
*   
*   @param name Field name (such as "rho", "u", etc.)
*   @return std::string string checkpoint filename
*/
__host__
std::string getCheckpointFilenameRead(
    std::string name
);

/**
*   @brief Get the checkpoint filename to write to
*   
*   @param name Field name (such as "rho", "u", etc.)
*   @return std::string string checkpoint filename
*/
__host__
std::string getCheckpointFilenameWrite(
    std::string name
);


/**
*   @brief Operation over checkpoint, save or load 
*   
*   @param oper operation to do, either __LOAD_CHECKPOINT or __SAVE_CHECKPOINT 
*   @param fMom Populations array
*   @param fGhost.X_0: populations to be pulled when threadIdx.x == NX-1
*   @param fGhost.X_1: populations to be pulled when threadIdx.x == 0
*   @param fGhost.Y_0: populations to be pulled when threadIdx.y == NY-1
*   @param fGhost.Y_1: populations to be pulled when threadIdx.y == 0
*   @param fGhost.Z_0: populations to be pulled when threadIdx.z == NZ-1
*   @param fGhost.Z_1: populations to be pulled when threadIdx.z == 0
*   @param step Pointer to current step value in main
*/
__host__
void operateSimCheckpoint( 
    int oper,
    dfloat* fMom,
    ghostData fGhost,
    #ifdef SECOND_DIST 
    ghostData g_fGhost,
    #endif
    int* step
);


/**
*   @brief Load simulation checkpoint
*
*   @param fMom Populations array
*   @param fGhost.X_0: populations to be pulled when threadIdx.x == NX-1
*   @param fGhost.X_1: populations to be pulled when threadIdx.x == 0
*   @param fGhost.Y_0: populations to be pulled when threadIdx.y == NY-1
*   @param fGhost.Y_1: populations to be pulled when threadIdx.y == 0
*   @param fGhost.Z_0: populations to be pulled when threadIdx.z == NZ-1
*   @param fGhost.Z_1: populations to be pulled when threadIdx.z == 0
*   @param step Pointer to current step value in main
*/
__host__
void loadSimCheckpoint( 
    dfloat* fMom,
    ghostData fGhost,
    #ifdef SECOND_DIST 
    ghostData g_fGhost,
    #endif
    int *step
);


/**
*   @brief Save simulation checkpoint
*
*   @param fMom Populations array
*   @param fGhost.X_0: populations to be pulled when threadIdx.x == NX-1
*   @param fGhost.X_1: populations to be pulled when threadIdx.x == 0
*   @param fGhost.Y_0: populations to be pulled when threadIdx.y == NY-1
*   @param fGhost.Y_1: populations to be pulled when threadIdx.y == 0
*   @param fGhost.Z_0: populations to be pulled when threadIdx.z == NZ-1
*   @param fGhost.Z_1: populations to be pulled when threadIdx.z == 0
*   @param step Pointer to current step value in main
*/
__host__
void saveSimCheckpoint( 
    dfloat* fMom,
    ghostData fGhost,
    #ifdef SECOND_DIST 
    ghostData g_fGhost,
    #endif
    int *step
);



















#endif //!__CHECKPOINT_H