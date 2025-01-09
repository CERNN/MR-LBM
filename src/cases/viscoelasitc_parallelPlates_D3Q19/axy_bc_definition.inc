switch (nodeType){
    case BULK:
        break;
    case NORTH:
        AxyVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH:
        AxyVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Axy_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
