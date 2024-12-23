#ifdef A_XX_DIST
dfloat AxxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif
#ifdef A_XY_DIST
dfloat AxyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif
#ifdef A_XZ_DIST
dfloat AxzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif
#ifdef A_YY_DIST
dfloat AyyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif
#ifdef A_YZ_DIST
dfloat AyzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif
#ifdef A_ZZ_DIST
dfloat AzzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
#endif

//#include "includeFiles\conformationTransport\conformation_gradient.inc"

#ifdef A_XX_DIST
#include "conformationTransport\Axx.inc"
#endif
#ifdef A_XY_DIST
#include "conformationTransport\Axy.inc"
#endif
#ifdef A_XZ_DIST
#include "conformationTransport\Axz.inc"
#endif
#ifdef A_YY_DIST
#include "conformationTransport\Ayy.inc"
#endif
#ifdef A_YZ_DIST
#include "conformationTransport\Ayz.inc"
#endif
#ifdef A_ZZ_DIST
#include "conformationTransport\Azz.inc"
#endif

//compute r  c
//r  c
/*
dfloat AGU_GUTA_xx = 2*AxxVar*duxdx_t30 + 2*AxyVar*duydx_t30 + 2*AxzVar*duzdx_t30;
dfloat AGU_GUTA_yy = 2*AxyVar*duxdy_t30 + 2*AyyVar*duydy_t30 + 2*AyzVar*duzdy_t30;
dfloat AGU_GUTA_zz = 2*AxzVar*duxdz_t30 + 2*AyzVar*duydz_t30 + 2*AzzVar*duzdz_t30;

dfloat AGU_GUTA_xy = AxxVar*duxdy_t30 + AxyVar*duxdx_t30 + AxyVar*duydy_t30 + AyyVar*duydx_t30 + AxzVar*duzdy_t30 + AyzVar*duzdx_t30;
dfloat AGU_GUTA_xz = AxxVar*duxdz_t30 + AxzVar*duxdx_t30 + AxyVar*duydz_t30 + AyzVar*duydx_t30 + AxzVar*duzdz_t30 + AzzVar*duzdx_t30;
dfloat AGU_GUTA_yz = AxyVar*duxdz_t30 + AxzVar*duxdy_t30 + AyyVar*duydz_t30 + AyzVar*duydy_t30 + AyzVar*duzdz_t30 + AzzVar*duzdy_t30;


dfloat trA = AxxVar + AyyVar + AzzVar;
dfloat aa = -inv_fenep_lambda/(1-trA/(fenep_re*fenep_re));
dfloat bb = -inv_fenep_lambda/(1-3.0/(fenep_re*fenep_re));


dfloat conformation_G_xx = aa * AxxVar - bb + AGU_GUTA_xx/F_M_I_SCALE;
dfloat conformation_G_yy = aa * AxxVar - bb + AGU_GUTA_yy/F_M_I_SCALE;
dfloat conformation_G_zz = aa * AxxVar - bb + AGU_GUTA_zz/F_M_I_SCALE;

dfloat conformation_G_xy = aa * AxxVar + AGU_GUTA_xy/F_M_I_SCALE;
dfloat conformation_G_xz = aa * AxxVar + AGU_GUTA_xz/F_M_I_SCALE;
dfloat conformation_G_yz = aa * AxxVar + AGU_GUTA_yz/F_M_I_SCALE;

//Source term for evolution
AxxVar = AxxVar + conformation_G_xx;
AxyVar = AxyVar + conformation_G_xy;
AxzVar = AxzVar + conformation_G_xz;
AyyVar = AyyVar + conformation_G_yy;
AyzVar = AyzVar + conformation_G_yz;
AzzVar = AzzVar + conformation_G_zz;


// Save neighboring halo cells in shared memory
#include "includeFiles\conformationTransport\confSave.inc"


//compute force for mometum equation
//F = div * (mu/lambda)*(A - I)
//F = (mu/lambda) * div(A);
//todo COMPUTE GRADIENT OF CONFORMATION VECTOR

L_Fx += (dAxxdx + dAxydy + dAxzdz)*(nu_p*inv_fenep_lambda) * 0.0; //zero just to debug for now
L_Fy += (dAxydx + dAyydy + dAyzdz)*(nu_p*inv_fenep_lambda) * 0.0; //zero just to debug for now
L_Fz += (dAxzdx + dAyzdy + dAzzdz)*(nu_p*inv_fenep_lambda) * 0.0; //zero just to debug for now

*/
#ifdef A_XX_DIST
//xx
// direct reconstruction since difussion is zero
#include COLREC_AXX_RECONSTRUCTION
//store population in halo
#include "includeFiles\conformationTransport\popSave_Axx.inc"
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxxVar;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qx_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qy_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XX_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axx_qz_t30;
#endif
#ifdef A_XY_DIST
//xy
// direct reconstruction since difussion is zero
#include COLREC_AXY_RECONSTRUCTION
//store population in halo
#include "includeFiles\conformationTransport\popSave_Axy.inc"
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxyVar;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qx_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qy_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axy_qz_t30;
#endif
#ifdef A_XZ_DIST
//xz
// direct reconstruction since difussion is zero
#include COLREC_AXZ_RECONSTRUCTION
//store population in halo
#include "includeFiles\conformationTransport\popSave_Axz.inc"
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AxzVar;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qx_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qy_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_XZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Axz_qz_t30;
#endif
#ifdef A_YY_DIST
//yy
// direct reconstruction since difussion is zero
#include COLREC_AYY_RECONSTRUCTION
//store population in halo
#include "includeFiles\conformationTransport\popSave_Ayy.inc"
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AyyVar;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qx_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qy_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YY_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayy_qz_t30;
#endif
#ifdef A_YZ_DIST
//yz
// direct reconstruction since difussion is zero
#include COLREC_AYZ_RECONSTRUCTION
//store population in halo
#include "includeFiles\conformationTransport\popSave_Ayz.inc"
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AyzVar;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qx_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qy_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_YZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Ayz_qz_t30;
#endif
#ifdef A_ZZ_DIST
//zz
// direct reconstruction since difussion is zero
#include COLREC_AZZ_RECONSTRUCTION
//store population in halo
#include "includeFiles\conformationTransport\popSave_Azz.inc"
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = AzzVar;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qx_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qy_t30;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, A_ZZ_CZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = Azz_qz_t30;
#endif