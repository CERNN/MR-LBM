//dfloat cc;
//dfloat sgw = (gW1 + 4.0*gW2);
switch (nodeType){
    case BULK:
        break;
    case EAST: //COLD
        cVar =  T_COLD;
        qx_t30   = fMom[idxMom(threadIdx.x-1, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = 0;
        qz_t30   = 0;
        break;
    case WEST: //HOT
        cVar =  T_HOT;
        qx_t30   = fMom[idxMom(threadIdx.x+1, threadIdx.y, threadIdx.z, M2_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qy_t30   = 0;
        qz_t30   = 0;
        break;
    case NORTH:
            //free-slip bc for thermal IS NOT bounce back

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y-1, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;
        break;
    case SOUTH:
            //free-slip bc for thermal IS NOT bounce back

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = fMom[idxMom(threadIdx.x, threadIdx.y+1, threadIdx.z, M2_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        qz_t30   = 0;
        break;
    case FRONT:

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = 0;
        qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z-1, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case BACK:

        cVar =  T_HOT - T_DELTA_T*((dfloat)x/(dfloat)NX);
        qx_t30   = 0;
        qy_t30   = 0;
        qz_t30   = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z+1, M2_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
        break;
    case NORTH:
            //free-slip bc for thermal IS NOT bounce back
            gNode[ 4] = gNode[ 3];
            gNode[13] = gNode[ 7];
            gNode[18] = gNode[11];
            gNode[ 8] = gNode[14];
            gNode[12] = gNode[17];
            cVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
            invC= 1.0/cVar;

            qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
            qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
            qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
        break;
    case SOUTH:
            //free-slip bc for thermal IS NOT bounce back
            gNode[ 3] = gNode[ 4];
            gNode[ 7] = gNode[13];
            gNode[11] = gNode[18];
            gNode[14] = gNode[ 8];
            gNode[17] = gNode[12];
            cVar = gNode[0] + gNode[1] + gNode[2] + gNode[3] + gNode[4] + gNode[5] + gNode[6] + gNode[7] + gNode[8] + gNode[9] + gNode[10] + gNode[11] + gNode[12] + gNode[13] + gNode[14] + gNode[15] + gNode[16] + gNode[17] + gNode[18];
            invC= 1.0/cVar;

            qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + gNode[7] - gNode[ 8] + gNode[ 9] - gNode[10] + gNode[13] - gNode[14] + gNode[15] - gNode[16]));
            qy_t30 = F_M_I_SCALE*((gNode[3] - gNode[4] + gNode[7] - gNode[ 8] + gNode[11] - gNode[12] + gNode[14] - gNode[13] + gNode[17] - gNode[18]));
            qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + gNode[11] - gNode[12] + gNode[16] - gNode[15] + gNode[18] - gNode[17]));
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
