//dfloat cc;
//dfloat sgw = (gW1 + 4.0*gW2);

switch (nodeType){
    case BULK:
        break;
    case EAST: //COLD
        cVar =  T_COLD;
        qx_t30   = fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = 0;//fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;//fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case WEST: //HOT
        cVar =  T_COLD;
        qx_t30   = fMom[idxMom(threadIdx.x+1, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = 0;//fMom[idxMom(threadIdx.x+1, threadIdx.y, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;//fMom[idxMom(threadIdx.x+1, threadIdx.y, threadIdx.z, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH: //ADIABATIC
        cVar =  fMom[idxMom(threadIdx.x, threadIdx.y+1, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;

    case NORTH: //ADIABATIC
        cVar =  fMom[idxMom(threadIdx.x, threadIdx.y-1, threadIdx.z, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case FRONT: //ADIABATIC
        cVar =  fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case BACK: //ADIABATIC
        cVar =  fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z+1, M2_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    default:
        break;
}