multiplyTerm = AxyVar * gW0;
pics2 = 1.0;

AxyNode[ 0] = multiplyTerm * (pics2);
multiplyTerm = AxyVar * gW1;
AxyNode[ 1] = multiplyTerm * (pics2 + ux_t30);// + udx_t30 );
AxyNode[ 2] = multiplyTerm * (pics2 - ux_t30);// - udx_t30 );
AxyNode[ 3] = multiplyTerm * (pics2 + uy_t30);// + udy_t30 );
AxyNode[ 4] = multiplyTerm * (pics2 - uy_t30);// - udy_t30 );
AxyNode[ 5] = multiplyTerm * (pics2 + uz_t30);// + udz_t30 );
AxyNode[ 6] = multiplyTerm * (pics2 - uz_t30);// - udz_t30 );
multiplyTerm = AxyVar * gW2;
AxyNode[ 7] = multiplyTerm * (pics2 + ux_t30 + uy_t30);// + udx_t30 + udy_t30 );
AxyNode[ 8] = multiplyTerm * (pics2 - ux_t30 - uy_t30);// - udx_t30 - udy_t30 );
AxyNode[ 9] = multiplyTerm * (pics2 + ux_t30 + uz_t30);// + udx_t30 + udz_t30 );
AxyNode[10] = multiplyTerm * (pics2 - ux_t30 - uz_t30);// - udx_t30 - udz_t30 );
AxyNode[11] = multiplyTerm * (pics2 + uy_t30 + uz_t30);// + udy_t30 + udz_t30 );
AxyNode[12] = multiplyTerm * (pics2 - uy_t30 - uz_t30);// - udy_t30 - udz_t30 );
AxyNode[13] = multiplyTerm * (pics2 + ux_t30 - uy_t30);// + udx_t30 - udy_t30 );
AxyNode[14] = multiplyTerm * (pics2 - ux_t30 + uy_t30);// - udx_t30 + udy_t30 );
AxyNode[15] = multiplyTerm * (pics2 + ux_t30 - uz_t30);// + udx_t30 - udz_t30 );
AxyNode[16] = multiplyTerm * (pics2 - ux_t30 + uz_t30);// - udx_t30 + udz_t30 );
AxyNode[17] = multiplyTerm * (pics2 + uy_t30 - uz_t30);// + udy_t30 - udz_t30 );
AxyNode[18] = multiplyTerm * (pics2 - uy_t30 + uz_t30);// - udy_t30 + udz_t30 );

