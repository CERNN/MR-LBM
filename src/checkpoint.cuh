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
#ifdef PARTICLE_MODEL
    #include "particles\class\particle.cuh"
#endif
#include <sys/stat.h>
#include <sys/types.h>



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
*   @brief Writes dfloat3SoA GPU arrays content into files
*
*   @param arr GPU arrays to read content from
*   @param foldername Foldername to save files to
*   @param arr_size_bytes Size in bytes to write for each file. If zero, reads whole file
*   @param tmp Temporary array used to read from file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__ 
void writeFilesIntoDfloat3SoA(
    dfloat3SoA arr, 
    const std::string foldername, 
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
*   @brief Reads files contents into dfloat3SoA GPU arrays
*
*   @param arr GPU arrays to write content to
*   @param foldername Foldername to read files from
*   @param arr_size_bytes Size in bytes to read for each file. 
*   @param tmp Temporary array used to write to file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__
void readFilesIntoDfloat3SoA(
    dfloat3SoA arr, 
    const std::string foldername, 
    const size_t arr_size_bytes, 
    void* tmp
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
*   @param ghostInterface interface block transfer information
*   @param step Pointer to current step value in main
*/
__host__
void operateSimCheckpoint( 
    int oper,
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int* step
);

#ifdef PARTICLE_MODEL
__host__
void operateSimCheckpoinT( 
    int oper,
    ParticlesSoA& particlesSoA,
    int* step
);
#endif


/**
*   @brief Load simulation checkpoint
*
*   @param fMom Populations array
*   @param ghostInterface interface block transfer information
*   @param step Pointer to current step value in main
*   @return 0 = fail to load checkpoint, 1 = load success;
*/
__host__
int loadSimCheckpoint( 
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int *step
);
#ifdef PARTICLE_MODEL
/**
*   @brief Load simulation checkpoint IBM
*
*   @param particlesSoA Particles
*   @param step Pointer to current step value in main
*   @return 0 = fail to load checkpoint, 1 = load success;
*/
__host__
int loadSimCheckpointParticle( 
    ParticlesSoA& particlesSoA,
    int *step
);
#endif


/**
*   @brief Save simulation checkpoint
*
*   @param fMom Populations array
*   @param ghostInterface interface block transfer information
*   @param step Pointer to current step value in main
*/
__host__
void saveSimCheckpoint( 
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int *step
);

#ifdef PARTICLE_MODEL

/**
*   @brief Save simulation checkpoint IMB
*
*   @param particlesSoA particle
*   @param step Pointer to current step value in main
*/
__host__
void saveSimCheckpointParticle( 
    ParticlesSoA& particlesSoA,
    int *step
);
#endif



#endif //!__CHECKPOINT_H