switch (nodeType){
    case BULK:
        break;
    case SOUTH:
        cVar = T_HOT;
        qx_t30   = 0;//fMom[idxMom(threadIdx.x, threadIdx.y+1, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y+1, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;//fMom[idxMom(threadIdx.x, threadIdx.y+1, threadIdx.z, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    case NORTH:
        cVar = T_COLD;
        qx_t30   = 0;//fMom[idxMom(threadIdx.x, threadIdx.y-1, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y-1, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;//fMom[idxMom(threadIdx.x, threadIdx.y-1, threadIdx.z, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    default:
        break;
}