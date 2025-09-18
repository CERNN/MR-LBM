switch (nodeType) {
    case BULK:
        break;
    case SOLID_NODE:
        break;
    default:
        dfloat3 unkNode = dfloat3(x,y,z);
        dfloat3 cCenter = dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,z);
        dfloat3 normal_vector = vector_normalize(cCenter - unkNode);

        dfloat3 node1 = unkNode + 1.0*normal_vector;
        dfloat3 node2 = unkNode + 2.0*normal_vector;
        dfloat3 node3 = unkNode + 3.0*normal_vector;

        AxxVar     = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XX_C_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XX_C_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XX_C_INDEX,fMom))*0.5;     
        Axx_qx_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XX_CX_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XX_CX_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XX_CX_INDEX,fMom))*0.5;
        Axx_qy_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XX_CY_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XX_CY_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XX_CY_INDEX,fMom))*0.5;
        Axx_qz_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XX_CZ_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XX_CZ_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XX_CZ_INDEX,fMom))*0.5;
                 
        break;
}
