multiplyTerm = AyyVar * gW0;
pics2 = 1.0;

ANode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AyyVar * gW1;
ANode[ 1] = multiplyTerm * (pics2 + ux_t30 + Ayy_udx_t30 );
ANode[ 2] = multiplyTerm * (pics2 - ux_t30 - Ayy_udx_t30 );
ANode[ 3] = multiplyTerm * (pics2 + uy_t30 + Ayy_udy_t30 );
ANode[ 4] = multiplyTerm * (pics2 - uy_t30 - Ayy_udy_t30 );
ANode[ 5] = multiplyTerm * (pics2 + uz_t30 + Ayy_udz_t30 );
ANode[ 6] = multiplyTerm * (pics2 - uz_t30 - Ayy_udz_t30 );
multiplyTerm = AyyVar * gW2;
ANode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + Ayy_udx_t30 + Ayy_udy_t30 );
ANode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - Ayy_udx_t30 - Ayy_udy_t30 );
ANode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30 + Ayy_udx_t30 + Ayy_udz_t30 );
ANode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30 - Ayy_udx_t30 - Ayy_udz_t30 );
ANode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30 + Ayy_udy_t30 + Ayy_udz_t30 );
ANode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30 - Ayy_udy_t30 - Ayy_udz_t30 );
ANode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + Ayy_udx_t30 - Ayy_udy_t30 );
ANode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - Ayy_udx_t30 + Ayy_udy_t30 );
ANode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30 + Ayy_udx_t30 - Ayy_udz_t30 );
ANode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30 - Ayy_udx_t30 + Ayy_udz_t30 );
ANode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30 + Ayy_udy_t30 - Ayy_udz_t30 );
ANode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30 - Ayy_udy_t30 + Ayy_udz_t30 );

