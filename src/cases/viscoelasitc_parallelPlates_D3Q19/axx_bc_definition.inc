switch (nodeType){
    case BULK:
        break;
    case NORTH:
        AxxVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH:
        AxxVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axx_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
