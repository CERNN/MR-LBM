switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AxxVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        
        Axx_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axx_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axx_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;

    case SOUTH:
        AxxVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axx_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axx_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;

        Axx_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                    - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                        + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) / 2.0;
        break;
    default:
        break;
}
