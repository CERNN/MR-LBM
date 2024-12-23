multiplyTerm = AxzVar * gW0;
pics2 = 1.0;

AxzNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AxzVar * gW1;
AxzNode[ 1] = multiplyTerm * (pics2 + ux_t30 + Axz_udx_t30 );
AxzNode[ 2] = multiplyTerm * (pics2 - ux_t30 - Axz_udx_t30 );
AxzNode[ 3] = multiplyTerm * (pics2 + uy_t30 + Axz_udy_t30 );
AxzNode[ 4] = multiplyTerm * (pics2 - uy_t30 - Axz_udy_t30 );
AxzNode[ 5] = multiplyTerm * (pics2 + uz_t30 + Axz_udz_t30 );
AxzNode[ 6] = multiplyTerm * (pics2 - uz_t30 - Axz_udz_t30 );
multiplyTerm = AxzVar * gW2;
AxzNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30 + Axz_udx_t30 + Axz_udy_t30);
AxzNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30 - Axz_udx_t30 - Axz_udy_t30);
AxzNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30 + Axz_udx_t30 + Axz_udz_t30);
AxzNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30 - Axz_udx_t30 - Axz_udz_t30);
AxzNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30 + Axz_udy_t30 + Axz_udz_t30);
AxzNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30 - Axz_udy_t30 - Axz_udz_t30);
AxzNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30 + Axz_udx_t30 - Axz_udy_t30);
AxzNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30 - Axz_udx_t30 + Axz_udy_t30);
AxzNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30 + Axz_udx_t30 - Axz_udz_t30);
AxzNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30 - Axz_udx_t30 + Axz_udz_t30);
AxzNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30 + Axz_udy_t30 - Axz_udz_t30);
AxzNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30 - Axz_udy_t30 + Axz_udz_t30);

