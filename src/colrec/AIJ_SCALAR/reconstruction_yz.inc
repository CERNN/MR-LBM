multiplyTerm = AyzVar * gW0;
pics2 = 1.0;

gNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AyzVar * gW1;
gNode[ 1] = multiplyTerm * (pics2 + ux_t30 + Ayz_udx_t30 );
gNode[ 2] = multiplyTerm * (pics2 - ux_t30 - Ayz_udx_t30 );
gNode[ 3] = multiplyTerm * (pics2 + uy_t30 + Ayz_udy_t30 );
gNode[ 4] = multiplyTerm * (pics2 - uy_t30 - Ayz_udy_t30 );
gNode[ 5] = multiplyTerm * (pics2 + uz_t30 + Ayz_udz_t30 );
gNode[ 6] = multiplyTerm * (pics2 - uz_t30 - Ayz_udz_t30 );
multiplyTerm = AyzVar * gW2;
gNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + Ayz_udx_t30 + Ayz_udy_t30 );
gNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - Ayz_udx_t30 - Ayz_udy_t30 );
gNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30 + Ayz_udx_t30 + Ayz_udz_t30 );
gNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30 - Ayz_udx_t30 - Ayz_udz_t30 );
gNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30 + Ayz_udy_t30 + Ayz_udz_t30 );
gNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30 - Ayz_udy_t30 - Ayz_udz_t30 );
gNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + Ayz_udx_t30 - Ayz_udy_t30 );
gNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - Ayz_udx_t30 + Ayz_udy_t30 );
gNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30 + Ayz_udx_t30 - Ayz_udz_t30 );
gNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30 - Ayz_udx_t30 + Ayz_udz_t30 );
gNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30 + Ayz_udy_t30 - Ayz_udz_t30 );
gNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30 - Ayz_udy_t30 + Ayz_udz_t30 );

