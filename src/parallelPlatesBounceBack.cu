#include "boundaryCondition.cuh"

__device__ void gpuBoundaryConditionMom(
    dim3 threadIdx, dim3 blockIdx,  dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz ){

    /*
    * uVar = 3u
    * pixx = 4.5pixx
    * pixy = 9.0pixy
    */

    dfloat rho_I;
    dfloat inv_rho_I;
    dfloat pixx_I;
    dfloat pixy_I;
    dfloat pixz_I;
    dfloat piyy_I;
    dfloat piyz_I;
    dfloat pizz_I;
    dfloat rho;
    dfloat inv_rho;
    dfloat multiplyTerm;;
    dfloat pics2;

    switch (dNodeType){
        case BULK:
            break;
        case SOUTH:
            uxVar = 0.0;  
            uyVar = 0.0;  
            uzVar = 0.0;  

            //A1 IO: 4/8/12/13/18
            rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17]; 
            inv_rho_I = 1.0 / rho_I; 
            //A2

            pixx_I = inv_rho_I *   (pop[1] + pop[2] + pop[7] + pop[9] + pop[10] + pop[14] + pop[15] + pop[16] -  cs2*rho_I);
            pixy_I = inv_rho_I *  ((pop[7]) - (pop[14]));
            pixz_I = inv_rho_I *  ((pop[9] + pop[10]) - (pop[15] + pop[16]));
            piyy_I = inv_rho_I *   (pop[3] +pop[7] + pop[11] + pop[14] + pop[17]  - cs2*rho_I);
            piyz_I = inv_rho_I *  ((pop[11])-(pop[17]));
            pizz_I = inv_rho_I *   (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[15] + pop[16] + pop[17]  - cs2*rho_I);

            //2.0*4.5 = 9.0
            rho = rho_I * ( 9.0 * T_OMEGA * (piyy_I) + 12.0)/(9.0 + OMEGA); //A34
            inv_rho = 1.0/rho;

            // 4.5 and 9.0 multiplication is because they are stored on that format
            piyy = inv_rho * (1.5 * rho_I * piyy_I + rho /6.0); //A35
            pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
            pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
            pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
            piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
            pixz = inv_rho * rho_I * pixz_I; //A38

            rhoVar = rho;

            break;

        case NORTH:

            uxVar = 0.0;  
            uyVar = 0.0;  
            uzVar = 0.0;  

            //A1 IO: 3/7/11/14/17
            rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
            inv_rho_I = 1.0 / rho_I;

            pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[8] + pop[9] + pop[10] + pop[13] + pop[15] + pop[16] -  cs2*rho_I);
            pixy_I = inv_rho_I * (( pop[ 8]) - (pop[13] ));
            pixz_I = inv_rho_I * ((pop[9] + pop[10]) - (pop[15] + pop[16]));
            piyy_I = inv_rho_I *  ( pop[4]  + pop[8] + pop[11] + pop[12] + pop[13] + pop[18] - cs2*rho_I);
            piyz_I = inv_rho_I * ((pop[12])-(pop[18]));
            pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[10] + pop[12] + pop[15] + pop[16]+ pop[18] - cs2*rho_I);

            //2.0*4.5 = 9.0
            rho = rho_I * ( 9.0 * T_OMEGA * (piyy_I) + 12.0)/(9.0 + OMEGA); //A34
            inv_rho = 1.0/rho;
            
            // 4.5 and 9.0 multiplication is because they are stored on that format
            piyy = inv_rho * (1.5 * rho_I * piyy_I + rho /6.0); //A35
            pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
            pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
            pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
            piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
            pixz = inv_rho * rho_I * pixz_I; //A38

            rhoVar = rho;
            
            break;
        default:
            break;
    }
}

__global__ void gpuInitialization_nodeType(
    char *dNodeType)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
    char nodeType;

    if (y == 0){ //S
        nodeType = SOUTH;
    }else if (y == (NY - 1)){ // N
        nodeType = NORTH;
    }else{
        nodeType = BULK;
    }
    
    dNodeType[idxNodeType(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;

}