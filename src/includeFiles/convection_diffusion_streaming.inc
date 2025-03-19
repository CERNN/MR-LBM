//overwrite values
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  0)] = gNode[ 1];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  1)] = gNode[ 2];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  2)] = gNode[ 3];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  3)] = gNode[ 4];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  4)] = gNode[ 5];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  5)] = gNode[ 6];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  6)] = gNode[ 7];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  7)] = gNode[ 8];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  8)] = gNode[ 9];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z,  9)] = gNode[10];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 10)] = gNode[11];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 11)] = gNode[12];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 12)] = gNode[13];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 13)] = gNode[14];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 14)] = gNode[15];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 15)] = gNode[16];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 16)] = gNode[17];
s_pop[idxPopBlock(threadIdx.x, threadIdx.y, threadIdx.z, 17)] = gNode[18];


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