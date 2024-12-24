#ifndef __MLBM_VISCOELASTIC_H
#define __MLBM_VISCOELASTIC_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "var.h"
#include "includeFiles/interface.h"
#include "boundaryCondition.cuh"

#ifdef A_XX_DIST
__global__ void gpuConformationXXCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save);
#endif

#ifdef A_XY_DIST
__global__ void gpuConformationXYCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save);
#endif

#ifdef A_XZ_DIST
__global__ void gpuConformationXZCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save);
#endif

#ifdef A_YY_DIST
__global__ void gpuConformationYYCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save);
#endif

#ifdef A_YZ_DIST
__global__ void gpuConformationYZCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save);
#endif

#ifdef A_ZZ_DIST
__global__ void gpuConformationZZCollisionStream(
    dfloat *fMom, unsigned int *dNodeType, ghostInterfaceData ghostInterface,
    unsigned int step,
    bool save);
#endif

#endif __MLBM_H