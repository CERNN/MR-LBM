
#include "auxFunctions.cuh"


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
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem  //
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
    unsigned int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * (blockIdx.z));

    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem  //
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


__host__
void mean_moment(dfloat *fMom, dfloat *meanMom, int m_index, size_t step){

    dfloat* sum;
    cudaMalloc((void**)&sum, NUM_BLOCK * sizeof(dfloat));

    int nt_x = BLOCK_NX;
    int nt_y = BLOCK_NY;
    int nt_z = BLOCK_NZ;
    int nb_x = NX / nt_x;
    int nb_y = NX / nt_y;
    int nb_z = NX / nt_z;

    sumReductionThread << <dim3(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_BLOCK_Z), dim3(BLOCK_NX, BLOCK_NY, BLOCK_NZ) >> > (fMom, sum,m_index);

    nb_x = NUM_BLOCK_X;
    nb_y = NUM_BLOCK_Y;
    nb_z = NUM_BLOCK_Z;

    int current_block_size = nb_x * nb_y * nb_z;
   
    while (true) {
        current_block_size = nb_x * nb_y * nb_z;
        if (current_block_size <= BLOCK_LBM_SIZE) { // last reduction
            sumReductionBlock << <1, dim3(nb_x, nb_y, nb_z) >> > (sum, sum);
            break;
        }
        else {
            nb_x = (nb_x < BLOCK_NX ? 1 : nb_x / BLOCK_NX);
            nb_y = (nb_y < BLOCK_NY ? 1 : nb_y / BLOCK_NY);
            nb_z = (nb_z < BLOCK_NZ ? 1 : nb_z / BLOCK_NZ);
            if (nb_x * nb_y * nb_z * nt_x * nt_y * nt_z > current_block_size) {
                if (nb_x > nb_y && nb_x > nb_z)
                    nt_x /= 2;
                else if (nb_y > nb_x && nb_y > nb_z)
                    nt_y /= 2;
                else
                    nt_z /= 2;
            }
            sumReductionBlock << <dim3(nb_x, nb_y, nb_z), dim3(nt_x, nt_y, nt_z) >> > (sum, sum);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    dfloat temp;
    
    checkCudaErrors(cudaMemcpy(&temp, sum, sizeof(dfloat), cudaMemcpyDeviceToHost)); 

    temp = (temp/(dfloat)NUMBER_LBM_NODES)- 1.0;   // TODO: NO IDEA WHY DOUBLING  
    printf("step %d temp %e \n ",step, temp);
    checkCudaErrors(cudaMemcpy(meanMom, &temp, sizeof(dfloat), cudaMemcpyHostToDevice)); 
    cudaFree(sum);
    
}


