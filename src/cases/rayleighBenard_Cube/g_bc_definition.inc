//dfloat cc;
//dfloat sgw = (gW1 + 4.0*gW2);
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
    case NORTH:
            //free-slip bc for thermal IS NOT bounce back

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y-1, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;
        break;
    case SOUTH:
            //free-slip bc for thermal IS NOT bounce back

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y+1, threadIdx.z, M_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;
        break;
    case FRONT:

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = 0;
        qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case BACK:

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = 0;
        qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z+1, M_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case SOUTH_WEST:
        cVar =  T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case NORTH_WEST:
        cVar = T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case SOUTH_EAST:
        cVar =  T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case NORTH_EAST:
        cVar =  T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case SOUTH_WEST_BACK:
        cVar =  T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case SOUTH_WEST_FRONT:
        cVar =  T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case NORTH_WEST_BACK:
        cVar = T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case NORTH_WEST_FRONT:
        cVar = T_HOT;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case SOUTH_EAST_BACK:
        cVar =  T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case SOUTH_EAST_FRONT:
        cVar =  T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case NORTH_EAST_BACK:
        cVar = T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    case NORTH_EAST_FRONT:
        cVar = T_COLD;
        qx_t30 = 0;
        qy_t30 = 0;
        qz_t30 = 0;
        break;
    default:
        break;
}
