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



#include "reduction.cuh"

__global__ 
void sumReductionThread(dfloat* g_idata, dfloat* g_odata, int m_index)
{
    #if (BLOCK_LBM_SIZE == 512)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 256)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 128)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 64)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 32)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 16)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 8)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #else
        extern __shared__ dfloat sdata[];
    #endif


    //global index in the array
    unsigned int i =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, m_index, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}

__global__ 
void sumReductionThread_TKE(dfloat* g_idata, dfloat* g_odata, dfloat *meanMom)
{
    #if (BLOCK_LBM_SIZE == 512)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 256)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 128)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 64)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 32)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 16)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 8)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #else
        extern __shared__ dfloat sdata[];
    #endif


    //global index in the array
    unsigned int ix =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iy =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iz =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    dfloat fluc_ux = (g_idata[ix] - meanMom[ix])*(g_idata[ix] - meanMom[ix]);
    dfloat fluc_uy = (g_idata[iy] - meanMom[iy])*(g_idata[iy] - meanMom[iy]);
    dfloat fluc_uz = (g_idata[iz] - meanMom[iz])*(g_idata[iz] - meanMom[iz]);

    sdata[tid] = (fluc_ux*fluc_ux + fluc_uy*fluc_uy + fluc_uz*fluc_uz)/2;
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}


__global__ 
void sumReductionThread_rho(dfloat* g_idata, dfloat* g_odata)
{
    #if (BLOCK_LBM_SIZE == 512)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 256)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 128)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 64)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 32)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 16)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 8)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #else
        extern __shared__ dfloat sdata[];
    #endif


    //global index in the array
    unsigned int ix =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = (g_idata[ix]);
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}

__global__ 
void sumReductionThread_KE(dfloat* g_idata, dfloat* g_odata)
{
    #if (BLOCK_LBM_SIZE == 512)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 256)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 128)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 64)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 32)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 16)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 8)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #else
        extern __shared__ dfloat sdata[];
    #endif

    //global index in the array
    unsigned int ix =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iy =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    unsigned int iz =  idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = (g_idata[ix]*g_idata[ix] + g_idata[iy]*g_idata[iy]+g_idata[iz]*g_idata[iz])/2;
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}


__global__ 
void sumReductionScalar(dfloat* g_idata, dfloat* g_odata)
{
    #if (BLOCK_LBM_SIZE == 512)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 256)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 128)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 64)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 32)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 16)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 8)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #else
        extern __shared__ dfloat sdata[];
    #endif

    //global index in the array
    unsigned int i =  idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}



__global__ 
void sumReductionBlock(dfloat* g_idata, dfloat* g_odata)
{
    #if (BLOCK_LBM_SIZE == 512)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 256)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 128)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 64)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 32)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 16)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #elif (BLOCK_LBM_SIZE == 8)
        __shared__ dfloat sdata[BLOCK_LBM_SIZE];
    #else
        extern __shared__ dfloat sdata[];
    #endif

    //global index in the array
    unsigned int i = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z + blockDim.z * ((blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z))))));
    //thread index in the array
    unsigned int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z));
    //block index
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[bid] = sdata[0];
    }
}