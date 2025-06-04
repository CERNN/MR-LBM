switch (nodeType) {
    case BULK:
        break;
    case SOLID_NODE:
        break;
    default:
        dfloat3 unkNode = dfloat3(x,y,z);
        dfloat3 cCenter = dfloat3(DUCT_CENTER_X,DUCT_CENTER_Y,z)
        dfloat3 normal_vector = cCenter - unkNode;
        normal_vector = normal_vector / vector_normalize(normal_vector);

        dfloat3 node1 = unkNode + 1.0*normal_vector;
        dfloat3 node2 = unkNode + 2.0*normal_vector;
        dfloat3 node3 = unkNode + 3.0*normal_vector;

        AxyVar     = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XY_C_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XY_C_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XY_C_INDEX,fMom))*0.5;     
        Axy_qx_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XY_CX_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XY_CX_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XY_CX_INDEX,fMom))*0.5;
        Axy_qy_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XY_CY_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XY_CY_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XY_CY_INDEX,fMom))*0.5;
        Axy_qz_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_XY_CZ_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_XY_CZ_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_XY_CZ_INDEX,fMom))*0.5;
                 
        break;
}
