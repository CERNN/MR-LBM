/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For additional information on the license terms, see the CUDA EULA at
https://docs.nvidia.com/cuda/eula/index.html

*/


#ifndef __REDUCTION_CUH
#define __REDUCTION_CUH

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "var.h"
#include "nodeTypeMap.h"


/**
 *  @brief Perform paralel reduction of all threads of a block of a moment
 *  @param g_idata: moment point
 *  @param g_odata: sum of moments inside the block
 *  @param m_index: moment index
*/
__global__ 
void sumReductionThread(dfloat* g_idata, dfloat* g_odata, int m_index);

/**
 *  @brief Perform paralel reduction of all threads of a block of for kinetic energy
 *  @param g_idata: moment point
 *  @param g_odata: sum of moments inside the block
*/
__global__ 
void sumReductionThread_rho(dfloat* g_idata, dfloat* g_odata);

/**
 *  @brief Perform paralel reduction of all threads of a block of for kinetic energy
 *  @param g_idata: moment point
 *  @param g_odata: sum of moments inside the block
*/
__global__ 
void sumReductionThread_KE(dfloat* g_idata, dfloat* g_odata);

#ifdef CONVECTION_DIFFUSION_TRANSPORT
#ifdef CONFORMATION_TENSOR
/**
 *  @brief Perform paralel reduction of all threads of a block of for spring energy
 *  @param g_idata: moment point
 *  @param g_odata: sum of moments inside the block
*/
__global__ 
void sumReductionThread_SE(dfloat* g_idata, dfloat* g_odata);
#endif
#endif

/**
 *  @brief Perform paralel reduction of all threads of a block of for kinetic energy
 *  @param g_idata: moment point
 *  @param g_odata: sum of moments inside the block
 *  @param m_fMom: mean moment array
*/
__global__ 
void sumReductionThread_TKE(dfloat* g_idata, dfloat* g_odata, dfloat *m_fMom);

/**
 *  @brief Perform paralel reduction of all threads of a block of for scalar array
 *  @param g_idata: moment point
 *  @param g_odata: sum of moments inside the block
*/
__global__ 
void sumReductionScalar(dfloat* g_idata, dfloat* g_odata);

/**
 *  @brief Perform paralel reduction of reduced block point
 *  @param g_idata: reduced pointer
 *  @param g_odata: reduced pointer
*/
__global__ 
void sumReductionBlock(dfloat* g_idata, dfloat* g_odata);




#endif // !__REDUCTION_CUH