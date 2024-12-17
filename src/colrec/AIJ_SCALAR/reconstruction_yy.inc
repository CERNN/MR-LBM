multiplyTerm = AyyVar * gW0;
pics2 = 1.0;

AyyNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AyyVar * gW1;
AyyNode[ 1] = multiplyTerm * (pics2 + ux_t30);// + udx_t30 );
AyyNode[ 2] = multiplyTerm * (pics2 - ux_t30);// - udx_t30 );
AyyNode[ 3] = multiplyTerm * (pics2 + uy_t30);// + udy_t30 );
AyyNode[ 4] = multiplyTerm * (pics2 - uy_t30);// - udy_t30 );
AyyNode[ 5] = multiplyTerm * (pics2 + uz_t30);// + udz_t30 );
AyyNode[ 6] = multiplyTerm * (pics2 - uz_t30);// - udz_t30 );
multiplyTerm = AyyVar * gW2;
AyyNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30);// + udx_t30 + udy_t30 );
AyyNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30);// - udx_t30 - udy_t30 );
AyyNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30);// + udx_t30 + udz_t30 );
AyyNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30);// - udx_t30 - udz_t30 );
AyyNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30);// + udy_t30 + udz_t30 );
AyyNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30);// - udy_t30 - udz_t30 );
AyyNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30);// + udx_t30 - udy_t30 );
AyyNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30);// - udx_t30 + udy_t30 );
AyyNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30);// + udx_t30 - udz_t30 );
AyyNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30);// - udx_t30 + udz_t30 );
AyyNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30);// + udy_t30 - udz_t30 );
AyyNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30);// - udy_t30 + udz_t30 );

