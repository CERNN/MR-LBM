#ifndef __BC_MOMENT_D3Q19_H
#define __BC_MOMENT_D3Q19_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <cuda_runtime.h>
#include "../var.h"

__device__ void 
gpuBCMomentN(dfloat *pop, dfloat &rhoVar, char dNodeType,
             dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
             dfloat &pixx, dfloat &pixy, dfloat &pixz,
             dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentS(dfloat *pop, dfloat &rhoVar, char dNodeType,
             dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
             dfloat &pixx, dfloat &pixy, dfloat &pixz,
             dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentW(dfloat *pop, dfloat &rhoVar, char dNodeType,
             dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
             dfloat &pixx, dfloat &pixy, dfloat &pixz,
             dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentE(dfloat *pop, dfloat &rhoVar, char dNodeType,
             dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
             dfloat &pixx, dfloat &pixy, dfloat &pixz,
             dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentF(dfloat *pop, dfloat &rhoVar, char dNodeType,
             dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
             dfloat &pixx, dfloat &pixy, dfloat &pixz,
             dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentB(dfloat *pop, dfloat &rhoVar, char dNodeType,
             dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
             dfloat &pixx, dfloat &pixy, dfloat &pixz,
             dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentNW(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);
              
__device__ void 
gpuBCMomentNE(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentNF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentNB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSW(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSE(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentWF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentWB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentEF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentEB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentNWF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);
__device__ void 
gpuBCMomentNWB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentNEF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentNEB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSWF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSWB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSEF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);

__device__ void 
gpuBCMomentSEB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz);
#endif