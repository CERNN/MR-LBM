//overwrite values
s_pop[baseIdxPop +  0*BLOCK_LBM_SIZE] = gNode[ 1];
s_pop[baseIdxPop +  1*BLOCK_LBM_SIZE] = gNode[ 2];
s_pop[baseIdxPop +  2*BLOCK_LBM_SIZE] = gNode[ 3];
s_pop[baseIdxPop +  3*BLOCK_LBM_SIZE] = gNode[ 4];
s_pop[baseIdxPop +  4*BLOCK_LBM_SIZE] = gNode[ 5];
s_pop[baseIdxPop +  5*BLOCK_LBM_SIZE] = gNode[ 6];
s_pop[baseIdxPop +  6*BLOCK_LBM_SIZE] = gNode[ 7];
s_pop[baseIdxPop +  7*BLOCK_LBM_SIZE] = gNode[ 8];
s_pop[baseIdxPop +  8*BLOCK_LBM_SIZE] = gNode[ 9];
s_pop[baseIdxPop +  9*BLOCK_LBM_SIZE] = gNode[10];
s_pop[baseIdxPop + 10*BLOCK_LBM_SIZE] = gNode[11];
s_pop[baseIdxPop + 11*BLOCK_LBM_SIZE] = gNode[12];
s_pop[baseIdxPop + 12*BLOCK_LBM_SIZE] = gNode[13];
s_pop[baseIdxPop + 13*BLOCK_LBM_SIZE] = gNode[14];
s_pop[baseIdxPop + 14*BLOCK_LBM_SIZE] = gNode[15];
s_pop[baseIdxPop + 15*BLOCK_LBM_SIZE] = gNode[16];
s_pop[baseIdxPop + 16*BLOCK_LBM_SIZE] = gNode[17];
s_pop[baseIdxPop + 17*BLOCK_LBM_SIZE] = gNode[18];


//sync threads of the block so all populations are saved
__syncthreads();

/* pull */
gNode[ 1] = s_pop[idxPopBlock(xm1, threadIdx.y, threadIdx.z, 0)];
gNode[ 2] = s_pop[idxPopBlock(xp1, threadIdx.y, threadIdx.z, 1)];
gNode[ 3] = s_pop[idxPopBlock(threadIdx.x, ym1, threadIdx.z, 2)];
gNode[ 4] = s_pop[idxPopBlock(threadIdx.x, yp1, threadIdx.z, 3)];
gNode[ 5] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zm1, 4)];
gNode[ 6] = s_pop[idxPopBlock(threadIdx.x, threadIdx.y, zp1, 5)];
gNode[ 7] = s_pop[idxPopBlock(xm1, ym1, threadIdx.z, 6)];
gNode[ 8] = s_pop[idxPopBlock(xp1, yp1, threadIdx.z, 7)];
gNode[ 9] = s_pop[idxPopBlock(xm1, threadIdx.y, zm1, 8)];
gNode[10] = s_pop[idxPopBlock(xp1, threadIdx.y, zp1, 9)];
gNode[11] = s_pop[idxPopBlock(threadIdx.x, ym1, zm1, 10)];
gNode[12] = s_pop[idxPopBlock(threadIdx.x, yp1, zp1, 11)];
gNode[13] = s_pop[idxPopBlock(xm1, yp1, threadIdx.z, 12)];
gNode[14] = s_pop[idxPopBlock(xp1, ym1, threadIdx.z, 13)];
gNode[15] = s_pop[idxPopBlock(xm1, threadIdx.y, zp1, 14)];
gNode[16] = s_pop[idxPopBlock(xp1, threadIdx.y, zm1, 15)];
gNode[17] = s_pop[idxPopBlock(threadIdx.x, ym1, zp1, 16)];
gNode[18] = s_pop[idxPopBlock(threadIdx.x, yp1, zm1, 17)];