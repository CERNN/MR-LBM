#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionMom(
    dim3 threadIdx, dim3 blockIdx,  dfloat pop[Q], dfloat& rhoVar, 
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz ){
    
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    /*
    * uVar = 3u
    * pixx = 4.5pixx
    * pixy = 9.0pixy
    */

    if (y == 0) // S
    {

        uxVar = 0.0;  
        uyVar = 0.0;  
        uzVar = 0.0;  

        //2.0*4.5 = 9.0
        dfloat rho = rhoVar * (12.0 + 2.0 * T_OMEGA * (piyy))/(9.0 + OMEGA);
        dfloat inv_rho =1.0/rho;

        piyy = inv_rho * (1.5 * rhoVar * piyy + rho*0.75); // 4.5/6.0 = 0.75
        pixx = inv_rho * (4.0/33.0) * rhoVar * (10.0*pixx-pizz);
        pizz = inv_rho * (4.0/33.0) * rhoVar * (10.0*pizz-pixx);
        pixy = inv_rho * 2.0*rhoVar*pixy;
        piyz = inv_rho * 2.0*rhoVar*piyz;        
        pixz = inv_rho * rhoVar*pixz;

        rhoVar = rho;
    }
    else if (y == (NY - 1)) // N
    {
        uxVar = 0.0;  
        uyVar = 0.0;  
        uzVar = 0.0;  

        //2.0*4.5 = 9.0
        dfloat rho = rhoVar * (12.0 + 2.0 * T_OMEGA * (piyy))/(9.0 + OMEGA);
        dfloat inv_rho =1.0/rho;
        
        piyy = inv_rho * (1.5 * rhoVar * piyy + rho*0.75); // 4.5/6.0 = 0.75
        pixx = inv_rho * (4.0/33.0) * rhoVar * (10.0*pixx-pizz);
        pizz = inv_rho * (4.0/33.0) * rhoVar * (10.0*pizz-pixx);
        pixy = inv_rho * 2.0*rhoVar*pixy;
        piyz = inv_rho * 2.0*rhoVar*piyz;        
        pixz = inv_rho * rhoVar*pixz;

        rhoVar = rho;
    }
    else if (x == 0) // W
    {
    }
    else if (x == (NX - 1)) // E
    {
    }
    else if (z == 0) // B
    {
    }
    else if (z == (NZ_TOTAL-1)) // F
    {  
    }

}