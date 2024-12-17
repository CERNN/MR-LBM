multiplyTerm = AxxVar * gW0;
pics2 = 1.0;

AxxNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AxxVar * gW1;
AxxNode[ 1] = multiplyTerm * (pics2 + ux_t30);// + udx_t30 );
AxxNode[ 2] = multiplyTerm * (pics2 - ux_t30);// - udx_t30 );
AxxNode[ 3] = multiplyTerm * (pics2 + uy_t30);// + udy_t30 );
AxxNode[ 4] = multiplyTerm * (pics2 - uy_t30);// - udy_t30 );
AxxNode[ 5] = multiplyTerm * (pics2 + uz_t30);// + udz_t30 );
AxxNode[ 6] = multiplyTerm * (pics2 - uz_t30);// - udz_t30 );
multiplyTerm = AxxVar * gW2;
AxxNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30);// + udx_t30 + udy_t30 );
AxxNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30);// - udx_t30 - udy_t30 );
AxxNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30);// + udx_t30 + udz_t30 );
AxxNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30);// - udx_t30 - udz_t30 );
AxxNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30);// + udy_t30 + udz_t30 );
AxxNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30);// - udy_t30 - udz_t30 );
AxxNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30);// + udx_t30 - udy_t30 );
AxxNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30);// - udx_t30 + udy_t30 );
AxxNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30);// + udx_t30 - udz_t30 );
AxxNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30);// - udx_t30 + udz_t30 );
AxxNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30);// + udy_t30 - udz_t30 );
AxxNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30);// - udy_t30 + udz_t30 );

