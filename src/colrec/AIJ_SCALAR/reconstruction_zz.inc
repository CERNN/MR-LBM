multiplyTerm = AzzVar * gW0;
pics2 = 1.0;

AzzNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AzzVar * gW1;
AzzNode[ 1] = multiplyTerm * (pics2 + ux_t30 + Azz_udx_t30 );
AzzNode[ 2] = multiplyTerm * (pics2 - ux_t30 - Azz_udx_t30 );
AzzNode[ 3] = multiplyTerm * (pics2 + uy_t30 + Azz_udy_t30 );
AzzNode[ 4] = multiplyTerm * (pics2 - uy_t30 - Azz_udy_t30 );
AzzNode[ 5] = multiplyTerm * (pics2 + uz_t30 + Azz_udz_t30 );
AzzNode[ 6] = multiplyTerm * (pics2 - uz_t30 - Azz_udz_t30 );
multiplyTerm = AzzVar * gW2;
AzzNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + Azz_udx_t30 + Azz_udy_t30 );
AzzNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - Azz_udx_t30 - Azz_udy_t30 );
AzzNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30 + Azz_udx_t30 + Azz_udz_t30 );
AzzNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30 - Azz_udx_t30 - Azz_udz_t30 );
AzzNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30 + Azz_udy_t30 + Azz_udz_t30 );
AzzNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30 - Azz_udy_t30 - Azz_udz_t30 );
AzzNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + Azz_udx_t30 - Azz_udy_t30 );
AzzNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - Azz_udx_t30 + Azz_udy_t30 );
AzzNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30 + Azz_udx_t30 - Azz_udz_t30 );
AzzNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30 - Azz_udx_t30 + Azz_udz_t30 );
AzzNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30 + Azz_udy_t30 - Azz_udz_t30 );
AzzNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30 - Azz_udy_t30 + Azz_udz_t30 );

