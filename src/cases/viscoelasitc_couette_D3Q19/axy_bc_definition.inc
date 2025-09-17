switch (nodeType) {
    case BULK:
        break;
    case NORTH:
            AxyVar = gNode[0] + gNode[1] + gNode[2] + 2*gNode[3] + gNode[5] + gNode[6] + 2*gNode[7] + gNode[9] + gNode[10] + 2*gNode[11] + 2*gNode[14] + gNode[15] + gNode[16] + 2*gNode[17];

            Axy_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2] + 2*gNode[7] + gNode[ 9] - gNode[10] - 2*gNode[14] + gNode[15] - gNode[16]));
            Axy_qy_t30 = 0;
            Axy_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] + 2*gNode[11] + gNode[16] - gNode[15] - 2*gNode[17]));
        break;
    case SOUTH:
            AxyVar = gNode[0] + gNode[1] + gNode[2] + 2*gNode[4] + gNode[5] + gNode[6] + 2*gNode[8] + gNode[9] + gNode[10] + 2*gNode[12] + 2*gNode[13] + gNode[15] + gNode[16] + 2*gNode[18];

            Axy_qx_t30 = F_M_I_SCALE*((gNode[1] - gNode[2]- 2*gNode[ 8] + gNode[ 9] - gNode[10] + 2*gNode[13] + gNode[15] - gNode[16]));
            Axy_qy_t30 = 0;
            Axy_qz_t30 = F_M_I_SCALE*((gNode[5] - gNode[6] + gNode[9] - gNode[10] - 2*gNode[12] + gNode[16] - gNode[15] + 2*gNode[18]));
        break;
    default:
        break;
}


