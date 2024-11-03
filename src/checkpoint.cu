/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br
*/

#include "checkpoint.cuh"


void createFolder(
    std::string foldername
){
    // Check if folder exists
    struct stat buffer;
    if(stat(foldername.c_str(), &buffer) == 0)
        return;
    #ifdef _WIN32
    std::string cmd = "md ";
    cmd += foldername;
    system(cmd.c_str());
    #else
    if(std::mkdir(foldername, 0777) == -1)
        std::cout << "Error creating folder '" << foldername << "'.\n";
    #endif
}



size_t getFileSize(
    std::string filename
){
    std::streampos fsize = 0;
    std::ifstream file( filename, std::ios::binary );

    fsize = file.tellg();
    file.seekg( 0, std::ios::end);
    fsize = file.tellg() - fsize;
    file.close();

    return fsize;
}

__host__
void readFileIntoArray(
    void* arr, 
    std::string filename, 
    size_t arr_size_bytes, 
    void* tmp
){
    FILE* file = fopen((filename+".bin").c_str(), "rb");
    // Check if file exists
    if(file == nullptr){
        std::cout << "Error reading file '" << filename << ".bin'. Exiting\n";
    }
    // load file size into array, if it is zero
    if(arr_size_bytes == 0){
        arr_size_bytes = getFileSize(filename);
    }

    // Read file into temporary array
    fread(tmp, arr_size_bytes, 1, file);
    // Copy file content in tmp to GPU array
    checkCudaErrors(cudaMemcpy(arr, tmp, arr_size_bytes, cudaMemcpyDefault));

    fclose(file);
}

__host__
void writeFileIntoArray(void* arr, const std::string filename, const size_t arr_size_bytes, void* tmp){
    FILE* file = fopen((filename+".bin").c_str(), "wb");
    // Check if file exists
    if(file == nullptr){
        std::cout << "Error opening file '" << filename << ".bin' to write. Exiting\n";
    }

    // Copy file content from GPU array to tmp
    checkCudaErrors(cudaMemcpy(tmp, arr, arr_size_bytes, cudaMemcpyDefault));

    // Write temporary array into file
    fwrite(tmp, arr_size_bytes, 1, file);

    fclose(file);
}

__host__
std::string getCheckpointFilenameRead(std::string name){
    std::string filename = SIMULATION_FOLDER_LOAD_CHECKPOINT;
    #ifdef _WIN32
    return filename + "\\\\" + ID_SIM + "\\\\checkpoint\\\\" +
        "_" + name;
    #else
    return filename + "/" + ID_SIM + "/checkpoint/" + 
        "_" + name;
    #endif
}

__host__
std::string getCheckpointFilenameWrite(std::string name){
    std::string filename = PATH_FILES;
    #ifdef _WIN32
    return filename + "\\\\" + ID_SIM + "\\\\checkpoint\\\\" + 
        "_" + name;
    #else
    return filename + "/" + ID_SIM + "/checkpoint/" + 
        "_" + name;
    #endif
}

__host__
void operateSimCheckpoint( 
    int oper,
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int* step
    )
{
    // Defining what functions to use (read or write to files)
    void (*f_arr)(void*, const std::string, size_t, void*);
    std::string (*f_filename)(std::string);

    if(oper == __LOAD_CHECKPOINT){
        f_arr = &readFileIntoArray;
        f_filename = &getCheckpointFilenameRead;
    }else if(oper == __SAVE_CHECKPOINT){
        f_arr = &writeFileIntoArray;
        f_filename = &getCheckpointFilenameWrite;
    }else{
        std::cout << "Invalid operation. Exiting\n";
        exit(-1);
    }

    // Everything will fit in this array
    dfloat* tmp = (dfloat*)malloc(MEM_SIZE_MOM);

    // Load/save current step
    f_arr(step, f_filename("curr_step"), sizeof(int), tmp);
    printf("Loaded checkpoint: step %d \n",step[0]);

    checkCudaErrors(cudaSetDevice(GPU_INDEX));
    // Load/save pop
    f_arr(fMom, f_filename("fMom"), MEM_SIZE_MOM, tmp);
    printf("Loaded checkpoint: moments \n");
    // Load/save auxilary populations
    f_arr(ghostInterface.fGhost.X_0, f_filename("fGhost.X_0"), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, tmp);
    f_arr(ghostInterface.fGhost.X_1, f_filename("fGhost.X_1"), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * QF, tmp);
    f_arr(ghostInterface.fGhost.Y_0, f_filename("fGhost.Y_0"), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, tmp);
    f_arr(ghostInterface.fGhost.Y_1, f_filename("fGhost.Y_1"), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * QF, tmp);
    f_arr(ghostInterface.fGhost.Z_0, f_filename("fGhost.Z_0"), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, tmp);
    f_arr(ghostInterface.fGhost.Z_1, f_filename("fGhost.Z_1"), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * QF, tmp);
    printf("Loaded checkpoint: f_pops \n");

    #ifdef SECOND_DIST 
    f_arr(ghostInterface.g_fGhost.X_0, f_filename("g_fGhost.X_0"), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.g_fGhost.X_1, f_filename("g_fGhost.X_1"), sizeof(dfloat) * NUMBER_GHOST_FACE_YZ * GF, tmp);
    f_arr(ghostInterface.g_fGhost.Y_0, f_filename("g_fGhost.Y_0"), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.g_fGhost.Y_1, f_filename("g_fGhost.Y_1"), sizeof(dfloat) * NUMBER_GHOST_FACE_XZ * GF, tmp);
    f_arr(ghostInterface.g_fGhost.Z_0, f_filename("g_fGhost.Z_0"), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, tmp);
    f_arr(ghostInterface.g_fGhost.Z_1, f_filename("g_fGhost.Z_1"), sizeof(dfloat) * NUMBER_GHOST_FACE_XY * GF, tmp);
    printf("Loaded checkpoint: g_pop \n");
    #endif

    free(tmp);
}


__host__
void loadSimCheckpoint( 
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int *step
    ){
    operateSimCheckpoint(__LOAD_CHECKPOINT, fMom, ghostInterface,step);
}

__host__
void saveSimCheckpoint( 
    dfloat* fMom,
    ghostInterfaceData ghostInterface,
    int *step
    ){
    std::string foldername = PATH_FILES; 
    foldername += "\\\\";
    foldername += ID_SIM;
    foldername += "\\\\checkpoint";
    createFolder(foldername);
    operateSimCheckpoint(__SAVE_CHECKPOINT, fMom,ghostInterface, step);
}