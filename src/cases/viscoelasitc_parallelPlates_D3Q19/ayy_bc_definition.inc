switch (nodeType){
    case BULK:
        break;
    case NORTH:
        AyyVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayy_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayy_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayy_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH:
        AyyVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayy_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayy_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Ayy_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
