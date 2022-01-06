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

    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (piyy_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;
    
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

    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (piyy_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;

    piyy = inv_rho * (1.5 * rho_I * piyy_I + rho /6.0); //A35
    pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
    pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
    pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
    piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
    pixz = inv_rho * rho_I * pixz_I; //A38

    rhoVar = rho;

}


__device__
void gpuBCMomentW( dfloat* pop, dfloat& rhoVar, char dNodeType, // x = 0
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //1/7/9/13/15
    dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[11] + pop[12]+ pop[14] + pop[16] + pop[17] + pop[18];
    dfloat inv_rho_I = 1 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * (( pop[ 8]) - ( pop[14]));
    dfloat pixz_I = inv_rho_I * (( pop[10]) - ( pop[16]));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[8] + pop[11] + pop[12] + pop[14] + pop[17] + pop[18] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11]+pop[12])-(pop[17]+pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[10] + pop[11] + pop[12] + pop[16] + pop[17] + pop[18] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pixx_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;

    pixx = inv_rho * (1.5 * rho_I * pixx_I + rho /6.0); //A35
    piyy = inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pizz_I); //A36
    pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - piyy_I); //A39
    pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
    pixz = inv_rho * 2.0 * rho_I * pixz_I; //A40       
    piyz = inv_rho * rho_I * piyz_I; //A38

    rhoVar = rho;

}

__device__
void gpuBCMomentE( dfloat* pop, dfloat& rhoVar, char dNodeType, //x = NX
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //2/8//10/14/16
    dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[12] + pop[13] + pop[15] + pop[17] + pop[18];
    dfloat inv_rho_I = 1 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[9] + pop[13] + pop[15]  - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * ((pop[7] ) - (pop[13] ));
    dfloat pixz_I = inv_rho_I * ((pop[9] ) - (pop[15] ));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[11] + pop[12] + pop[13] + pop[17] + pop[18] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11]+pop[12])-(pop[17]+pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[15] + pop[17] + pop[18] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pixx_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;

    pixx = inv_rho * (1.5 * rho_I * pixx_I + rho /6.0); //A35
    piyy = inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pizz_I); //A36
    pizz = inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - piyy_I); //A39
    pixy = inv_rho * 2.0 * rho_I * pixy_I; //A37
    pixz = inv_rho * 2.0 * rho_I * pixz_I; //A40       
    piyz = inv_rho * rho_I * piyz_I; //A38

    rhoVar = rho;

}

__device__
void gpuBCMomentB( dfloat* pop, dfloat& rhoVar, char dNodeType, // z = 0
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //5/9/11/16/18
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[6] + pop[7] + pop[8] + pop[10] + pop[12] + pop[13] + pop[14] + pop[15] + pop[17];
    dfloat inv_rho_I = 1 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[8] + pop[10] + pop[13] + pop[14] + pop[15] - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * ((pop[7] + pop[ 8]) - (pop[13] + pop[14]));
    dfloat pixz_I = inv_rho_I * (( pop[10]) - (pop[15] ));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[8] + pop[12] + pop[13] + pop[14] + pop[17] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12])-(pop[17]));
    dfloat pizz_I = inv_rho_I *  ( pop[6] + pop[10] + pop[12] + pop[15]  + pop[17] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pizz_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;

    pizz = inv_rho * (1.5 * rho_I * pizz_I + rho /6.0); //A35
    piyy = inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pixx_I); //A36
    pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - piyy_I); //A39
    pixz = inv_rho * 2.0 * rho_I * pixz_I; //A37
    piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
    pixy = inv_rho * rho_I * pixy_I; //A38

    rhoVar = rho;

}

__device__
void gpuBCMomentF( dfloat* pop, dfloat& rhoVar, char dNodeType, //z = NZ
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //6/10/12/15/17
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[7] + pop[8] + pop[9] + pop[11] + pop[13] + pop[14] + pop[16] + pop[18];
    dfloat inv_rho_I = 1 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[13] + pop[14] + pop[16] - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * ((pop[7] + pop[ 8]) - (pop[13] + pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[9] ) - (+ pop[16]));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[13] + pop[14] + pop[18] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11] )-(pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[9] + pop[11] + pop[16] + pop[18] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pizz_I) + 12.0)/(9.0 + OMEGA); //A34
    dfloat inv_rho = 1.0/rho;

    pizz = inv_rho * (1.5 * rho_I * pizz_I + rho /6.0); //A35
    piyy = inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pixx_I); //A36
    pixx = inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - piyy_I); //A39
    pixz = inv_rho * 2.0 * rho_I * pixz_I; //A37
    piyz = inv_rho * 2.0 * rho_I * piyz_I; //A40       
    pixy = inv_rho * rho_I * pixy_I; //A38

    rhoVar = rho;

}

