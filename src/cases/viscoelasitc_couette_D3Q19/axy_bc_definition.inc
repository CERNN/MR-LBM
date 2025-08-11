switch (nodeType) {
    case BULK:
        break;
    case NORTH:
        AxyVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        
        Axy_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axy_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axy_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y - 1, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y - 2, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y - 3, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        break;

    case SOUTH:
        AxyVar     = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axy_qx_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axy_qy_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;

        Axy_qz_t30 = (5 * fMom[idxMom(threadIdx.x, threadIdx.y + 1, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      - 4 * fMom[idxMom(threadIdx.x, threadIdx.y + 2, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]
                      + fMom[idxMom(threadIdx.x, threadIdx.y + 3, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)]) *0.5;
        break;

    default:
        break;
}
