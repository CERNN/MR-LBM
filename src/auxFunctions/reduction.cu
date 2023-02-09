#include "reduction.cuh"


__global__ 
void sumReductionThread(dfloat* g_idata, dfloat* g_odata, int m_index)
{
    #if (BLOCK_LBM_SIZE == 512)
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
void sumReductionBlock(dfloat* g_idata, dfloat* g_odata)
{
    #if (BLOCK_LBM_SIZE == 512)
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