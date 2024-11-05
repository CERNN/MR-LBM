//dfloat cc;
//dfloat sgw = (gW1 + 4.0*gW2);
dfloat c_I;
dfloat inv_c;

switch (nodeType){
    case BULK:
        break;
    case EAST: //COLD
        cVar =  T_COLD;
        qx_t30   = fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = 0;
        qz_t30   = 0;
        break;
    case WEST: //HOT
        cVar =  T_HOT;
        qx_t30   = fMom[idxMom(threadIdx.x+1, threadIdx.y, threadIdx.z, M_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = 0;
        qz_t30   = 0;
        break;
    default:
        break;
}


