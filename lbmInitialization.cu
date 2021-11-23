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
    dfloat P = N / (4.0 * M_PI);

    ux = U_MAX * sin(-2.0 * M_PI / 3.0) * cos(x / P) * sin(y / P) * cos(z / P) * 2.0 / sqrt(3.0);
    uy = U_MAX * sin(-2.0 * M_PI / 3.0) * cos(x / P) * sin(y / P) * cos(z / P) * 2.0 / sqrt(3.0);
    uz = 0.0;
    
    //ux =  U_MAX*sin(-2.0*M_PI/3.0)*cos(x/P)*sin(y/P)*sin(z/P)*2.0/sqrt(3.0);
    //uy = -U_MAX*sin(-2.0*M_PI/3.0)*sin(x/P)*cos(y/P)*sin(z/P)*2.0/sqrt(3.0);
    //uz = 0.0;
    
    // zeroth moment
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)] = RHO_0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)] = ux;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)] = uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)] = uz;

    //second moments
    //define equilibrium populations
    dfloat pop[Q];
    char c1, c2;
    for (int i = 0; i < Q; i++)
    {
        pop[i] = gpu_f_eq(w[i] * RHO_0,
                          3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1 - 1.5 * (ux * ux + uy * uy + uz * uz));
    }
    
    dfloat pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] - RHO_0 * cs2) / RHO_0;
    dfloat pixy = ((pop[7] + pop[8]) - (pop[13] + pop[14])) / RHO_0;
    dfloat pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) / RHO_0;
    dfloat piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] - RHO_0 * cs2) / RHO_0;
    dfloat piyz = ((pop[11] + pop[12]) - (pop[17] + pop[18])) / RHO_0;
    dfloat pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] - RHO_0 * cs2) / RHO_0;

    //pixx = pixx + OMEGA * (RHO_0 * ux * ux -  pixx)  + TT_OMEGA * (FX * ux + FX * ux);
    //pixy = pixy + OMEGA * (RHO_0 * ux * uy -  pixy)  + TT_OMEGA * (FX * uy + FY * ux);
    //pixz = pixz + OMEGA * (RHO_0 * ux * uz -  pixz)  + TT_OMEGA * (FX * uz + FZ * ux);
    //piyy = piyy + OMEGA * (RHO_0 * uy * uy -  piyy)  + TT_OMEGA * (FY * uy + FY * uy);
    //piyz = piyz + OMEGA * (RHO_0 * uy * uz -  piyz)  + TT_OMEGA * (FY * uz + FZ * uy);
    //pizz = pizz + OMEGA * (RHO_0 * uz * uz -  pizz)  + TT_OMEGA * (FZ * uz + FZ * uz);

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)] = pixx; //= RHO_0*ux*ux+RHO_0*cs2;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)] = pixy; //= RHO_0*ux*uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)] = pixz; //= RHO_0*ux*uz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)] = piyy; //= RHO_0*uy*uy+RHO_0*cs2;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)] = piyz; //= RHO_0*uy*uz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)] = pizz; //= RHO_0*uz*uz+RHO_0*cs2;
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

    dfloat rhoVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 0, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 1, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 2, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 3, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixx = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 4, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 5, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 6, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 7, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 8, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pizz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, 9, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat pop[Q];
    #pragma unroll //equation 6
    for (int i = 0; i < Q; i++)
    {
        pop[i] = rhoVar * w[i] * (1 
        + as2 * (uxVar * cx[i] + uyVar * cy[i] + uzVar * cz[i]) 
        + 0.5 * as2 * as2 * (
            pixx * (cx[i] * cx[i] - cs2) + 
            pixy * (cx[i] * cy[i]) + 
            pixz * (cx[i] * cz[i]) + 
            piyy * (cy[i] * cy[i] - cs2) + 
            piyz * (cy[i] * cz[i]) + 
            pizz * (cz[i] * cz[i] - cs2))
        );
    }

    //gpuInterfacePushCentered(threadIdx, blockIdx, pop, fGhostX_0, fGhostX_1, fGhostY_0, fGhostY_1, fGhostZ_0, fGhostZ_1);
    gpuInterfacePushOffset(threadIdx, blockIdx, pop, fGhostX_0, fGhostX_1, fGhostY_0, fGhostY_1, fGhostZ_0, fGhostZ_1);
}
