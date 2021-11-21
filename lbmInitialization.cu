#include "lbmInitialization.cuh"

__global__ void gpuInitialization_mom(
    dfloat *fMom)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;


    size_t index = idxScalarGlobal(x, y, z);
    //printf("threadIdx.x % d threadIdx.y % d threadIdx.z % d  bix %d biy %d biz %d --  x: %d y: %d z: %d idx %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, x, y, z, index);

    //first moments
    dfloat ux, uy, uz;

    ux = 0.05;
    uy = 0.05;
    uz = 0.05;
    
    //Taylor Green
    dfloat P = N/(2.0*M_PI);

    ux = 0.1*sin(-2.0*M_PI/3.0)*cos(x/P)*sin(y/P)*cos(z/P)*2.0/sqrt(3.0);
    uy = 0.1*sin(-2.0*M_PI/3.0)*cos(x/P)*sin(y/P)*cos(z/P)*2.0/sqrt(3.0);
    uz = 0.0;
    


    // zeroth moment
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)] = ux;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)] = uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)] = uz;

    //second moments
    //define equilibrium populations
    //dfloat feq[Q];
    //dfloat meq[3][3] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    /*char c1, c2;
    for (int i = 0; i < Q; i++)
    {
        feq[i] = gpu_f_eq(w[i] * RHO_0,
                          3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1 - 1.5 * (ux * ux + uy * uy + uz * uz));
    }
    for (int i = 0; i < Q; i++)
    {
        for (int d1 = 0; d1 < 3; d1++)
        {
            if (d1 == 0)
            { //x
                c1 = cx[i];
            }
            if (d1 == 1)
            { //y
                c1 = cy[i];
            }
            if (d1 == 2)
            { //z
                c1 = cz[i];
            }
            //if (c1 == 0)
            //{
            //    continue;
            //}
            for (int d2 = 0; d2 < 3; d2++)
            {
                if (d2 == 0)
                { //x
                    c2 = cx[i];
                }
                if (d2 == 1)
                { //y
                    c2 = cy[i];
                }
                if (d2 == 2)
                { //z
                    c2 = cz[i];
                }
                meq[d1][d2] += feq[i] * c1 * c2;
            }
        }
    }*/
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0*ux*ux+RHO_0*cs2;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0*ux*uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0*ux*uz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0*uy*uy+RHO_0*cs2;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0*uy*uz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0*uz*uz+RHO_0*cs2;
}

__global__ void gpuInitialization_pop(
    dfloat *fMom,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);
    // zeroth moment

    dfloat rhoVar, uxVar, uyVar, uzVar;

    rhoVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
    uxVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
    uyVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
    uzVar  = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat pop[Q];

    // Calculate temporary variables
    const dfloat p1_muu15 = 1 - 1.5 * (uxVar * uxVar + uyVar * uyVar + uzVar * uzVar);
    const dfloat rhoW0 = rhoVar * W0;
    const dfloat rhoW1 = rhoVar * W1;
    const dfloat rhoW2 = rhoVar * W2;
    const dfloat W1t3d2 = W1 * 3.0 / 2.0;
    const dfloat W2t3d2 = W2 * 3.0 / 2.0;
    const dfloat W1t9d2 = W1t3d2 * 3.0;
    const dfloat W2t9d2 = W2t3d2 * 3.0;

#ifdef D3Q27
    const dfloat rhoW3 = rhoVar * W3;
    const dfloat W3t9d2 = W3 * 9 / 2;
#endif
    const dfloat ux3 = 3 * uxVar;
    const dfloat uy3 = 3 * uyVar;
    const dfloat uz3 = 3 * uzVar;

    // Calculate equilibrium pop
    pop[ 0] = gpu_f_eq(rhoW0, 0, p1_muu15);
    pop[ 1] = gpu_f_eq(rhoW1, ux3, p1_muu15);
    pop[ 2] = gpu_f_eq(rhoW1, -ux3, p1_muu15);
    pop[ 3] = gpu_f_eq(rhoW1, uy3, p1_muu15);
    pop[ 4] = gpu_f_eq(rhoW1, -uy3, p1_muu15);
    pop[ 5] = gpu_f_eq(rhoW1, uz3, p1_muu15);
    pop[ 6] = gpu_f_eq(rhoW1, -uz3, p1_muu15);
    pop[ 7] = gpu_f_eq(rhoW2, ux3 + uy3, p1_muu15);
    pop[ 8] = gpu_f_eq(rhoW2, -ux3 - uy3, p1_muu15);
    pop[ 9] = gpu_f_eq(rhoW2, ux3 + uz3, p1_muu15);
    pop[10] = gpu_f_eq(rhoW2, -ux3 - uz3, p1_muu15);
    pop[11] = gpu_f_eq(rhoW2, uy3 + uz3, p1_muu15);
    pop[12] = gpu_f_eq(rhoW2, -uy3 - uz3, p1_muu15);
    pop[13] = gpu_f_eq(rhoW2, ux3 - uy3, p1_muu15);
    pop[14] = gpu_f_eq(rhoW2, -ux3 + uy3, p1_muu15);
    pop[15] = gpu_f_eq(rhoW2, ux3 - uz3, p1_muu15);
    pop[16] = gpu_f_eq(rhoW2, -ux3 + uz3, p1_muu15);
    pop[17] = gpu_f_eq(rhoW2, uy3 - uz3, p1_muu15);
    pop[18] = gpu_f_eq(rhoW2, -uy3 + uz3, p1_muu15);
#ifdef D3Q27
    pop[19] = gpu_f_eq(rhoW3, ux3 + uy3 + uz3, p1_muu15);
    pop[20] = gpu_f_eq(rhoW3, -ux3 - uy3 - uz3, p1_muu15);
    pop[21] = gpu_f_eq(rhoW3, ux3 + uy3 - uz3, p1_muu15);
    pop[22] = gpu_f_eq(rhoW3, -ux3 - uy3 + uz3, p1_muu15);
    pop[23] = gpu_f_eq(rhoW3, ux3 - uy3 + uz3, p1_muu15);
    pop[24] = gpu_f_eq(rhoW3, -ux3 + uy3 - uz3, p1_muu15);
    pop[25] = gpu_f_eq(rhoW3, -ux3 + uy3 + uz3, p1_muu15);
    pop[26] = gpu_f_eq(rhoW3, ux3 - uy3 - uz3, p1_muu15);
#endif

gpuInterfaceSpread(threadIdx,blockIdx,pop,fGhostX_0, fGhostX_1,fGhostY_0,fGhostY_1,fGhostZ_0,fGhostZ_1);

}
