multiplyTerm = AyzVar * gW0;
pics2 = 1.0;

AyzNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AyzVar * gW1;
AyzNode[ 1] = multiplyTerm * (pics2 + ux_t30 + Ayz_udx_t30 );
AyzNode[ 2] = multiplyTerm * (pics2 - ux_t30 - Ayz_udx_t30 );
AyzNode[ 3] = multiplyTerm * (pics2 + uy_t30 + Ayz_udy_t30 );
AyzNode[ 4] = multiplyTerm * (pics2 - uy_t30 - Ayz_udy_t30 );
AyzNode[ 5] = multiplyTerm * (pics2 + uz_t30 + Ayz_udz_t30 );
AyzNode[ 6] = multiplyTerm * (pics2 - uz_t30 - Ayz_udz_t30 );
multiplyTerm = AyzVar * gW2;
AyzNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + Ayz_udx_t30 + Ayz_udy_t30 );
AyzNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - Ayz_udx_t30 - Ayz_udy_t30 );
AyzNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30 + Ayz_udx_t30 + Ayz_udz_t30 );
AyzNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30 - Ayz_udx_t30 - Ayz_udz_t30 );
AyzNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30 + Ayz_udy_t30 + Ayz_udz_t30 );
AyzNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30 - Ayz_udy_t30 - Ayz_udz_t30 );
AyzNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + Ayz_udx_t30 - Ayz_udy_t30 );
AyzNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - Ayz_udx_t30 + Ayz_udy_t30 );
AyzNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30 + Ayz_udx_t30 - Ayz_udz_t30 );
AyzNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30 - Ayz_udx_t30 + Ayz_udz_t30 );
AyzNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30 + Ayz_udy_t30 - Ayz_udz_t30 );
AyzNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30 - Ayz_udy_t30 + Ayz_udz_t30 );

