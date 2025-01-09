switch (nodeType){
    case BULK:
        break;
    case NORTH:
        AxzVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH:
        AxzVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axz_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
