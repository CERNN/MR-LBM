switch (nodeType){
    case BULK:
        break;
    case NORTH:
        AzzVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH:
        AzzVar     = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] -  fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qx_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qy_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        Azz_qz_t30 = 2 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] - fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    default:
        break;
}
