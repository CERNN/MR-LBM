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

        AzzVar     = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_ZZ_C_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_ZZ_C_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_ZZ_C_INDEX,fMom))*0.5;     
        Azz_qx_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_ZZ_CX_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_ZZ_CX_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_ZZ_CX_INDEX,fMom))*0.5;
        Azz_qy_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_ZZ_CY_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_ZZ_CY_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_ZZ_CY_INDEX,fMom))*0.5;
        Azz_qz_t30 = (5*mom_trilinear_interp(node1.x,node1.y,node1.z,A_ZZ_CZ_INDEX,fMom)
                     -4*mom_trilinear_interp(node2.x,node2.y,node2.z,A_ZZ_CZ_INDEX,fMom)
                     -1*mom_trilinear_interp(node3.x,node3.y,node3.z,A_ZZ_CZ_INDEX,fMom))*0.5;
                 

        //printf("normal vector %f %f %f",normal_vector.x,normal_vector.y,normal_vector.z );
        //printf("node x: %d y: %d z: %d node1 x: %f y: %f z: %f node2 x: %f y: %f z: %f node3 x: %f y: %f z: %f \n",x,y,z,node1.x,node1.y,node1.z,node2.x,node2.y,node2.z,node3.x,node3.y,node3.z);

//        printf("step %d A %f qx %f qy %f qz %f \n",step,AzzVar,Azz_qx_t30,Azz_qy_t30,Azz_qz_t30);
//printf("step %d Azz %f %f %f %f \n",step,AzzVar,mom_trilinear_interp(node1.x,node1.y,node1.z,A_ZZ_C_INDEX,fMom),mom_trilinear_interp(node2.x,node2.y,node2.z,A_ZZ_C_INDEX,fMom),mom_trilinear_interp(node3.x,node3.y,node3.z,A_ZZ_C_INDEX,fMom));
        break;
}
