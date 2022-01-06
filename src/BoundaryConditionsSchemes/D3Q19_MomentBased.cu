#include "D3Q19_MomentBased.cuh"


__device__
void gpuBCMomentN( dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //A1 IO: 3/7/11/14/17
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[8] + pop[9] + pop[10] + pop[13] + pop[15] + pop[16] -  cs2*rho_I);
    dfloat pixy_I = inv_rho_I * (( pop[ 8]) - (pop[13] ));
    dfloat pixz_I = inv_rho_I * ((pop[9] + pop[10]) - (pop[15] + pop[16]));
    dfloat piyy_I = inv_rho_I *  ( pop[4]  + pop[8] + pop[11] + pop[12] + pop[13] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12])-(pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[10] + pop[12] + pop[15] + pop[16]+ pop[18] - cs2*rho_I);

    //2.0*4.5 = 9.0
    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (piyy_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;
    
    // 4.5 and 9.0 multiplication is because they are stored on that format
    piyy = inv_rho * (1.5 * rho_I * piyy_I + rho /6.0); //A35
    pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
    pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
    pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
    piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
    pixz = inv_rho * rho_I * pixz_I; //A38

    rhoVar = rho;

}

__device__
void gpuBCMomentS( dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //A1 IO: 4/8/12/13/18
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17]; 
    dfloat inv_rho_I = 1.0 / rho_I; 
    //A2

    dfloat pixx_I = inv_rho_I *   (pop[1] + pop[2] + pop[7] + pop[9] + pop[10] + pop[14] + pop[15] + pop[16] -  cs2*rho_I);
    dfloat pixy_I = inv_rho_I *  ((pop[7]) - (pop[14]));
    dfloat pixz_I = inv_rho_I *  ((pop[9] + pop[10]) - (pop[15] + pop[16]));
    dfloat piyy_I = inv_rho_I *   (pop[3] +pop[7] + pop[11] + pop[14] + pop[17]  - cs2*rho_I);
    dfloat piyz_I = inv_rho_I *  ((pop[11])-(pop[17]));
    dfloat pizz_I = inv_rho_I *   (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[15] + pop[16] + pop[17]  - cs2*rho_I);

    //2.0*4.5 = 9.0
    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (piyy_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;

    // 4.5 and 9.0 multiplication is because they are stored on that format
    piyy = inv_rho * (1.5 * rho_I * piyy_I + rho /6.0); //A35
    pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
    pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
    pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
    piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
    pixz = inv_rho * rho_I * pixz_I; //A38

    rhoVar = rho;

}
