#include "D3Q19_MomentBased.cuh"


// FULL VECTORS
/*

    //IO: 
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13] + pop[14] + pop[15] + pop[16] + pop[17] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7] + pop[8]) - (pop[13] + pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[9] + pop[10]) - (pop[15] + pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11] + pop[12]) - (pop[17] + pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] - cs2*rho_I);

*/


//TODO:
/*
inv_rho =  rho_I / rho;

*/



//FACES 

//l1 = 1 // l2 = 0 // l3 = 0
//u1 = uy // u2  = ux // u3 == uz
__device__
void gpuBCMomentN( dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //IO: 3/7/11/14/17
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[9] + pop[10] + pop[12] + pop[13] + pop[15] + pop[16] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[8] + pop[9] + pop[10] + pop[13] + pop[15] + pop[16] -  cs2*rho_I);
    dfloat pixy_I = inv_rho_I * (( pop[ 8]) - (pop[13] ));
    dfloat pixz_I = inv_rho_I * ((pop[9] + pop[10]) - (pop[15] + pop[16]));
    dfloat piyy_I = inv_rho_I *  ( pop[4]  + pop[8]+ pop[12] + pop[13] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12])-(pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[10] + pop[12] + pop[15] + pop[16]+ pop[18] - cs2*rho_I);

    dfloat rho = rho_I * (9.0 * T_OMEGA * (piyy_I) + 12.0) / (OMEGA * (1.0 - 6.0 * uyVar * uyVar) - 3.0 * uyVar * OMEGA_P1 + 9.0); //A34
    dfloat inv_rho = 1.0/rho;

    piyy = 4.5 * inv_rho * (1.5 * rho_I * piyy_I + rho * (ONESIXTH - 0.5 * uyVar)); //A35
    pixx = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
    pizz = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
    pixy = 9.0 * inv_rho * (2.0 * rho_I * pixy_I - ONETHIRD*rho*uxVar); //A37
    piyz = 9.0 * inv_rho * (2.0 * rho_I * piyz_I - ONETHIRD*rho*uzVar); //A40       
    pixz = 9.0 * inv_rho * rho_I * pixz_I; //A38

    rhoVar = rho;

}


//l1 = -1 // l2 = 0 // l3 = 0
//u1 = uy // u2  = ux // u3 == uz
__device__
void gpuBCMomentS( dfloat* pop, dfloat& rhoVar, char dNodeType,
    dfloat &uxVar , dfloat &uyVar , dfloat& uzVar , 
    dfloat &pixx  , dfloat &pixy  , dfloat& pixz  , 
    dfloat &piyy  , dfloat &piyz  , dfloat& pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //IO: 4/8/12/13/18
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[10] + pop[11] + pop[14] + pop[15] + pop[16] + pop[17]; 
    dfloat inv_rho_I = 1.0 / rho_I; 

    dfloat pixx_I = inv_rho_I *   (pop[1] + pop[2] + pop[7] + pop[9] + pop[10] + pop[14] + pop[15] + pop[16] -  cs2*rho_I);
    dfloat pixy_I = inv_rho_I *  ((pop[7]) - (pop[14]));
    dfloat pixz_I = inv_rho_I *  ((pop[9] + pop[10]) - (pop[15] + pop[16]));
    dfloat piyy_I = inv_rho_I *   (pop[3] +pop[7] + pop[11] + pop[14] + pop[17]  - cs2*rho_I);
    dfloat piyz_I = inv_rho_I *  ((pop[11])-(pop[17]));
    dfloat pizz_I = inv_rho_I *   (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[15] + pop[16] + pop[17]  - cs2*rho_I);

    dfloat rho = rho_I * (9.0 * T_OMEGA * (piyy_I) + 12.0) / (OMEGA * (1.0 - 6.0 * uyVar * uyVar) + 3.0 * uyVar * OMEGA_P1 + 9.0); //A34
    dfloat inv_rho = 1.0/rho;

    piyy = 4.5 * inv_rho * (1.5 * rho_I * piyy_I + rho * (ONESIXTH + 0.5 * uyVar)); //A35
    pixx = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - pizz_I); //A36
    pizz = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - pixx_I); //A39
    pixy = 9.0 * inv_rho * (2.0 * rho_I * pixy_I + ONETHIRD*rho*uxVar); //A37
    piyz = 9.0 * inv_rho * (2.0 * rho_I * piyz_I + ONETHIRD*rho*uzVar); //A40      
    pixz = 9.0 * inv_rho * rho_I * pixz_I; //A38

    rhoVar = rho;

}


//l1 = -1 // l2 = 0 // l3 = 0
//u1 = ux // u2  = uy // u3 == uz
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
    dfloat inv_rho_I = 1.0 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[10] + pop[14] + pop[16] - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * (( pop[ 8]) - ( pop[14]));
    dfloat pixz_I = inv_rho_I * (( pop[10]) - ( pop[16]));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[8] + pop[11] + pop[12] + pop[14] + pop[17] + pop[18] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11]+pop[12])-(pop[17]+pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[10] + pop[11] + pop[12] + pop[16] + pop[17] + pop[18] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pixx_I) + 12.0) / (OMEGA * (1.0 - 6.0 * uxVar * uxVar) + 3.0 * uxVar * OMEGA_P1 + 9.0); //A34
    dfloat inv_rho = 1.0/rho;

    pixx = 4.5 * inv_rho * (1.5 * rho_I * pixx_I  + rho * (ONESIXTH + 0.5 * uxVar)); //A35
    piyy = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pizz_I); //A36
    pizz = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - piyy_I); //A39
    pixy = 9.0 * inv_rho * (2.0 * rho_I * pixy_I + ONETHIRD*rho*uyVar); //A37
    pixz = 9.0 * inv_rho * (2.0 * rho_I * pixz_I + ONETHIRD*rho*uzVar); //A40
    piyz = 9.0 * inv_rho * rho_I * piyz_I; //A38

    rhoVar = rho;

}


//l1 = 1 // l2 = 0 // l3 = 0
//u1 = ux // u2  = uy // u3 == uz
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
    dfloat inv_rho_I = 1.0 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[9] + pop[13] + pop[15]  - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * ((pop[7] ) - (pop[13] ));
    dfloat pixz_I = inv_rho_I * ((pop[9] ) - (pop[15] ));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[11] + pop[12] + pop[13] + pop[17] + pop[18] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11]+pop[12])-(pop[17]+pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[11] + pop[12] + pop[15] + pop[17] + pop[18] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pixx_I) + 12.0) / (OMEGA * (1.0 - 6.0 * uxVar * uxVar) - 3.0 * uyVar * OMEGA_P1 + 9.0); //A34
    dfloat inv_rho = 1.0/rho;

    pixx = 4.5 * inv_rho * (1.5 * rho_I * pixx_I  + rho * (ONESIXTH - 0.5 * uxVar)); //A35
    piyy = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pizz_I); //A36
    pizz = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pizz_I - piyy_I); //A39
    pixy = 9.0 * inv_rho * (2.0 * rho_I * pixy_I - ONETHIRD*rho*uyVar); //A37
    pixz = 9.0 * inv_rho * (2.0 * rho_I * pixz_I - ONETHIRD*rho*uzVar); //A40
    piyz = 9.0 * inv_rho * rho_I * piyz_I; //A38

    rhoVar = rho;

}


//l1 = -1 // l2 = 0 // l3 = 0
//u1 = uz // u2  = ux // u3 == uy
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
    dfloat inv_rho_I = 1.0 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[8] + pop[10] + pop[13] + pop[14] + pop[15] - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * ((pop[7] + pop[ 8]) - (pop[13] + pop[14]));
    dfloat pixz_I = inv_rho_I * (( pop[10]) - (pop[15] ));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[8] + pop[12] + pop[13] + pop[14] + pop[17] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12])-(pop[17]));
    dfloat pizz_I = inv_rho_I *  ( pop[6] + pop[10] + pop[12] + pop[15]  + pop[17] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pizz_I) + 12.0) / (OMEGA * (1.0 - 6.0 * uzVar * uzVar) + 3.0 * uzVar * OMEGA_P1 + 9.0); //A34
    dfloat inv_rho = 1.0/rho;

    pizz = 4.5 * inv_rho * (1.5 * rho_I * pizz_I  + rho * (ONESIXTH + 0.5 * uzVar)); //A35
    piyy = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pixx_I); //A36
    pixx = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - piyy_I); //A39
    pixz = 9.0 * inv_rho * (2.0 * rho_I * pixz_I + ONETHIRD*rho*uxVar); //A37
    piyz = 9.0 * inv_rho * (2.0 * rho_I * piyz_I + ONETHIRD*rho*uyVar); //A40
    pixy = 9.0 * inv_rho * rho_I * pixy_I; //A38

    rhoVar = rho;

}

//l1 = 1 // l2 = 0 // l3 = 0
//u1 = uz // u2  = ux // u3 == uy
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
    dfloat inv_rho_I = 1.0 / rho_I;

    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[13] + pop[14] + pop[16] - cs2* rho_I) ;
    dfloat pixy_I = inv_rho_I * ((pop[7] + pop[ 8]) - (pop[13] + pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[9] ) - (+ pop[16]));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[13] + pop[14] + pop[18] - cs2* rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11] )-(pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[9] + pop[11] + pop[16] + pop[18] - cs2* rho_I);


    dfloat rho = rho_I * ( 9.0 * T_OMEGA * (pizz_I) + 12.0)/ (OMEGA * (1.0 - 6.0 * uzVar * uzVar) - 3.0 * uzVar * OMEGA_P1 + 9.0); //A34
    dfloat inv_rho = 1.0/rho;

    pizz = 4.5 * inv_rho * (1.5 * rho_I * pizz_I  + rho * (ONESIXTH - 0.5 * uzVar)); //A35
    piyy = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * piyy_I - pixx_I); //A36
    pixx = 4.5 * inv_rho * (4.0/33.0) * rho_I * (10.0 * pixx_I - piyy_I); //A39
    pixz = 9.0 * inv_rho * (2.0 * rho_I * pixz_I - ONETHIRD*rho*uxVar); //A37
    piyz = 9.0 * inv_rho * (2.0 * rho_I * piyz_I - ONETHIRD*rho*uyVar); //A40
    pixy = 9.0 * inv_rho * rho_I * pixy_I; //A38

    rhoVar = rho;

}







//EDGES


//l1 = -1 // l2 = 1 // l3 = 0
//u1 = ux // u2  = uy // u3 == uz
__device__ void 
gpuBCMomentNW(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 1/4/7/8/9/12/13/15/18 
    dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[5] + pop[6] + pop[10] + pop[11] + pop[14] + pop[16] + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[10]  + pop[14]  + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ( - (pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[10]) - (pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[11] + pop[14] + pop[17] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11])-(pop[17]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[10] + pop[11] + pop[16] + pop[17] - cs2*rho_I); 

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * piyy_I 
                - 2.0*pizz_I + 19.0*pixy_I); //A28
    dfloat dE = 720.0 - 660.0 * (-uxVar + uyVar) + OMEGA * (430.0 - 30.0 * (-uxVar + uyVar) - 414.0*uxVar*uyVar
                -690.0 * (uxVar*uxVar + uyVar*uyVar) - 69.0 * uzVar*uzVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 

    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + piyy_I - 6.0*pizz_I + 34.0*pixy_I) - (2.0/69.0)*(-8.0 - 15.0*uxVar + 8.0*uyVar)); //A30
    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pixx_I - 6.0*pizz_I + 34.0*pixy_I) - (2.0/69.0)*(-8.0 + 15.0*uzVar - 8.0*uxVar)); //A30  
    pizz = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * pixx_I - 9.0*piyy_I + 54.0 * pizz_I - 30.0 * pixy_I) - (4.0/69.0) * (1.0- uxVar + uzVar)); //A31
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I + ONETHIRD * uzVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I - ONETHIRD * uzVar); //A33
    pixy = 9.0 * (inv_rho * (1.0/23.0) * rho_I * (-(-17.0*pixx_I - 17.0*piyy_I + 10.0*pizz_I) + 118.0*pixy_I) - (19.0/69.0)*(-1.0 + uxVar - uzVar)); //A32

    rhoVar = rho;                  
}

//l1 = 1 // l2 = 1 // l3 = 0
//u1 = ux // u2  = uy // u3 == uz            
__device__ void 
gpuBCMomentNE(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 2/4/8/10/12/13/14/16/18  
    dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[5] + pop[6] + pop[7] + pop[9] + pop[11] + pop[15]  + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[9] + pop[15] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7]));
    dfloat pixz_I = inv_rho_I * ((pop[9]) - (pop[15])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[7] + pop[11] + pop[17] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11])-(pop[17]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[11] + pop[15] + pop[17] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * piyy_I 
                - 2.0*pizz_I - 19.0*pixy_I); //A28
    dfloat dE = 720.0 - 660.0 * (uxVar + uyVar) + OMEGA * (430.0 - 30.0 * (uxVar + uyVar) + 414.0*uxVar*uyVar
                -690.0 * (uxVar*uxVar + uyVar*uyVar) - 69.0 * uzVar*uzVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 

    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + piyy_I - 6.0*pizz_I - 34.0*pixy_I) - (2.0/69.0)*(-8.0 + 15.0*uxVar + 8.0*uyVar)); //A30
    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pixx_I - 6.0*pizz_I - 34.0*pixy_I) - (2.0/69.0)*(-8.0 + 15.0*uzVar + 8.0*uxVar));//A30  
    pizz = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * pixx_I - 9.0*piyy_I + 54.0 * pizz_I + 30.0 * pixy_I) - (4.0/69.0) * (1.0+ uxVar + uzVar)); //A31
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I - ONETHIRD * uzVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I - ONETHIRD * uzVar); //A33
    pixy = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*pixx_I - 17.0*piyy_I + 10.0*pizz_I) + 118.0*pixy_I) - (19.0/69.0)*(1.0 + uxVar + uzVar)); //A32

    rhoVar = rho;  
         
}

//l1 = -1 // l2 = -1 // l3 = 0
//u1 = ux // u2  = uy // u3 == uz
__device__ void 
gpuBCMomentSW(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 1/3/7/9/11/13/14/15/17    
    dfloat rho_I = pop[0] + pop[2] + pop[4] + pop[5] + pop[6] + pop[8] + pop[10] + pop[12] + pop[16] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[10] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[ 8]));
    dfloat pixz_I = inv_rho_I * ((pop[10]) - (pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[8] + pop[12] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12]) - (pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[10] + pop[12] + pop[15] + pop[18] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * piyy_I 
                - 2.0*pizz_I - 19.0*pixy_I); //A28
    dfloat dE = 720.0 - 660.0 * (-uxVar - uyVar) + OMEGA * (430.0 - 30.0 * (-uxVar - uyVar) + 414.0*uxVar*uyVar
                -690.0 * (uxVar*uxVar + uyVar*uyVar) - 69.0 * uzVar*uzVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 

    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + piyy_I - 6.0*pizz_I - 34.0*pixy_I) - (2.0/69.0)*(-8.0 - 15.0*uxVar - 8.0*uyVar)); //A30
    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pixx_I - 6.0*pizz_I - 34.0*pixy_I) - (2.0/69.0)*(-8.0 - 15.0*uzVar - 8.0*uxVar)); //A30  
    pizz = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * pixx_I - 9.0*piyy_I + 54.0 * pizz_I + 30.0 * pixy_I) - (4.0/69.0) * (1.0- uxVar - uzVar)); //A31
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I + ONETHIRD * uzVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I + ONETHIRD * uzVar); //A33
    pixy = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*pixx_I - 17.0*piyy_I + 10.0*pizz_I) + 118.0*pixy_I) - (19.0/69.0)*(1.0 - uxVar - uzVar)); //A32

    rhoVar = rho;   
                
}

//l1 = 1 // l2 = -1 // l3 = 0
//u1 = ux // u2  = uy // u3 == uz
__device__ void 
gpuBCMomentSE(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 2/3/7/8/10/11/14/16/17 
    dfloat rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[6] + pop[9] + pop[12] + pop[13] + pop[15] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[9] + pop[13] + pop[15] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ( - (pop[13]));
    dfloat pixz_I = inv_rho_I * ((pop[9]) - (pop[15])) ;
    dfloat piyy_I = inv_rho_I *  ( pop[4] + pop[12] + pop[13] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12]) - (pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[6] + pop[9] + pop[12] + pop[15] + pop[18] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * piyy_I 
                - 2.0*pizz_I + 19.0*pixy_I); //A28
    dfloat dE = 720.0 - 660.0 * (uxVar - uyVar) + OMEGA * (430.0 - 30.0 * (uxVar - uyVar) - 414.0*uxVar*uyVar
                -690.0 * (uxVar*uxVar + uyVar*uyVar) - 69.0 * uzVar*uzVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;
    
    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + piyy_I - 6.0*pizz_I + 34.0*pixy_I) - (2.0/69.0)*(-8.0 + 15.0*uxVar - 8.0*uyVar)); //A30
    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pixx_I - 6.0*pizz_I + 34.0*pixy_I) - (2.0/69.0)*(-8.0 + 15.0*uzVar - 8.0*uxVar));//A30  
    pizz = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * pixx_I - 9.0*piyy_I + 54.0 * pizz_I + 30.0 * pixy_I) - (4.0/69.0) * (-1.0 + uxVar - uzVar)); //A31
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I - ONETHIRD * uzVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I + ONETHIRD * uzVar); //A33
    pixy = 9.0 * (inv_rho * (1.0/23.0) * rho_I * (-(-17.0*pixx_I - 17.0*piyy_I + 10.0*pizz_I) + 118.0*pixy_I) - (19.0/69.0)*(-1.0 - uxVar + uzVar)); //A32

    rhoVar = rho;                    
}






//l1 = 1 // l2 = 1 // l3 = 0
//u1 = uy // u2  = uz // u3 == ux
__device__ void 
gpuBCMomentNF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 4/6/8/10/12/13/15/17/18 
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[5] + pop[7] + pop[9] + pop[11] + pop[14] + pop[16];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[9] + pop[14] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7]) - ( pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[9] ) - ( pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[7] + pop[11] + pop[14] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[9] + pop[11] + pop[16] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * piyy_I + 8.0 * pizz_I 
                - 2.0*pixx_I - 19.0*piyz_I); //A28
    dfloat dE = 720.0 - 660.0 * (uyVar + uzVar) + OMEGA * (430.0 - 30.0 * (uyVar + uzVar) + 414.0*uxVar*uzVar
                -690.0 * (uyVar*uyVar + uzVar*uzVar) - 69.0 * uxVar*uxVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;


    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pizz_I - 6.0*pixx_I - 34.0*piyz_I) - (2.0/69.0)*(-8.0 + 15.0*uyVar + 8.0*uzVar)); //A30  
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + piyy_I - 6.0*pixx_I - 34.0*piyz_I) - (2.0/69.0)*(-8.0 + 15.0*uzVar + 8.0*uyVar)); //A30
    pixx = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * piyy_I - 9.0*pizz_I + 54.0 * pixx_I + 30.0 * piyz_I) - (4.0/69.0) * (1.0 + uyVar + uzVar)); //A31
    pixy = 9.0 * (inv_rho * 2.0 * pixz_I - ONETHIRD * uxVar); //A33
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I - ONETHIRD * uxVar); //A33
    piyz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*piyy_I - 17.0*pizz_I + 10.0*pixx_I) + 118.0*pixy_I) - (19.0/69.0)*(1.0 + uyVar + uzVar)); //A32

    rhoVar = rho;                
}

//l1 = 1 // l2 = -1 // l3 = 0
//u1 = uy // u2  = uz // u3 == ux
__device__ void 
gpuBCMomentNB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO:  4/5/8/9/11/12/13/16/18
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[3] + pop[6] + pop[7] + pop[10] + pop[14] + pop[15] + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[7] + pop[10] + pop[14] + pop[15] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7]) - (pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[10]) - (pop[15])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[7] + pop[14] + pop[17] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * (- (pop[17]));
    dfloat pizz_I = inv_rho_I *  (pop[6] + pop[10] + pop[15] + pop[17] - cs2*rho_I); 

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * piyy_I + 8.0 * pizz_I 
                - 2.0*pixx_I + 19.0*piyz_I); //A28
    dfloat dE = 720.0 - 660.0 * (uyVar - uzVar) + OMEGA * (430.0 - 30.0 * (uyVar - uzVar) - 414.0*uxVar*uzVar
                -690.0 * (uyVar*uyVar + uzVar*uzVar) - 69.0 * uxVar*uxVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;

    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pizz_I - 6.0*pixx_I + 34.0*piyz_I) - (2.0/69.0)*(-8.0 + 15.0*uyVar - 8.0*uzVar)); //A30  
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + piyy_I - 6.0*pixx_I + 34.0*piyz_I) - (2.0/69.0)*(-8.0 - 15.0*uzVar + 8.0*uyVar)); //A30
    pixx = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * piyy_I - 9.0*pizz_I + 54.0 * pixx_I - 30.0 * piyz_I) - (4.0/69.0) * (1.0 + uyVar - uzVar)); //A31
    pixy = 9.0 * (inv_rho * 2.0 * pixz_I - ONETHIRD * uxVar); //A33
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I + ONETHIRD * uxVar); //A33
    piyz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * (-(-17.0*piyy_I - 17.0*pizz_I + 10.0*pixx_I) + 118.0*pixy_I) - (19.0/69.0)*(-1.0 - uyVar + uzVar)); //A32

    rhoVar = rho;  

    
                
}

//l1 = -1 // l2 = 1 // l3 = 0
//u1 = uy // u2  = uz // u3 == ux
__device__ void 
gpuBCMomentSF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 3/6/7/10/11/12/14/15/17
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[5] + pop[8] + pop[9] + pop[13] + pop[16] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[8] + pop[9] + pop[13] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[8]) - (pop[13]));
    dfloat pixz_I = inv_rho_I * ((pop[9]) - (pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[8] + pop[13] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ( - ( pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5]  + pop[9] + pop[16] + pop[18] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * piyy_I + 8.0 * pizz_I 
                - 2.0*pixx_I + 19.0*piyz_I); //A28
    dfloat dE = 720.0 - 660.0 * (-uyVar + uzVar) + OMEGA * (430.0 - 30.0 * (-uyVar + uzVar) - 414.0*uxVar*uzVar
                -690.0 * (uyVar*uyVar + uzVar*uzVar) - 69.0 * uxVar*uxVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;

    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pizz_I - 6.0*pixx_I + 34.0*piyz_I) - (2.0/69.0)*(-8.0 - 15.0*uyVar + 8.0*uzVar));//A30  
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + piyy_I - 6.0*pixx_I + 34.0*piyz_I) - (2.0/69.0)*(-8.0 + 15.0*uzVar - 8.0*uyVar)); //A30
    pixx = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * piyy_I - 9.0*pizz_I + 54.0 * pixx_I - 30.0 * piyz_I) - (4.0/69.0) * (1.0 - uyVar + uzVar)); //A31
    pixy = 9.0 * (inv_rho * 2.0 * pixz_I + ONETHIRD * uxVar); //A33
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I - ONETHIRD * uxVar); //A33
    piyz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * (-(-17.0*piyy_I - 17.0*pizz_I + 10.0*pixx_I) + 118.0*pixy_I) - (19.0/69.0)*(-1.0 + uyVar - uzVar)); //A32

    rhoVar = rho;  
}

//l1 = -1 // l2 = -1 // l3 = 0
//u1 = uy // u2  = uz // u3 == ux
__device__ void 
gpuBCMomentSB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 3/5/7/9/11/14/16/17/18
    dfloat rho_I = pop[0] + pop[1] + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[13] + pop[15];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[2] + pop[8] + pop[10] + pop[13] + pop[15] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[8]) - (pop[13]));
    dfloat pixz_I = inv_rho_I * ((pop[10]) - (pop[15])) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[8] + pop[12] + pop[13] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12]));
    dfloat pizz_I = inv_rho_I *  (pop[6] + pop[10] + pop[12] + pop[15] - cs2*rho_I); 

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * piyy_I + 8.0 * pizz_I 
                - 2.0*pixx_I - 19.0*piyz_I); //A28
    dfloat dE = 720.0 - 660.0 * (-uyVar - uzVar) + OMEGA * (430.0 - 30.0 * (-uyVar - uzVar) + 414.0*uxVar*uzVar
                -690.0 * (uyVar*uyVar + uzVar*uzVar) - 69.0 * uxVar*uxVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;


    piyy = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * piyy_I + pizz_I - 6.0*pixx_I - 34.0*piyz_I) - (2.0/69.0)*(-8.0 - 15.0*uyVar - 8.0*uzVar));//A30  
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + piyy_I - 6.0*pixx_I - 34.0*piyz_I) - (2.0/69.0)*(-8.0 - 15.0*uzVar - 8.0*uyVar)); //A30
    pixx = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 * piyy_I - 9.0*pizz_I + 54.0 * pixx_I + 30.0 * piyz_I) - (4.0/69.0) * (1.0 - uyVar - uzVar)); //A31
    pixy = 9.0 * (inv_rho * 2.0 * pixz_I + ONETHIRD * uxVar); //A33
    pixz = 9.0 * (inv_rho * 2.0 * pixz_I + ONETHIRD * uxVar); //A33
    piyz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*piyy_I - 17.0*pizz_I + 10.0*pixx_I) + 118.0*pixy_I) - (19.0/69.0)*(1.0 - uyVar - uzVar)); //A32

    rhoVar = rho;  
                   
}



//l1 = -1 // l2 = 1 // l3 = 0
//u1 = ux // u2  = uz // u3 == uy
__device__ void 
gpuBCMomentWF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO: 1/6/7/9/10/12/13/15/17
    dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[5] + pop[8] + pop[11] + pop[14] + pop[16] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[14] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[ 8]) - (pop[14]));
    dfloat pixz_I = inv_rho_I * (- (pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[8] + pop[11] + pop[14] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11])-(pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[11]+ pop[16] + pop[18] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * pizz_I 
                - 2.0*piyy_I + 19.0*pixz_I); //A28
    dfloat dE = 720.0 - 660.0 * (uxVar - uzVar) + OMEGA * (430.0 + 30.0 * (uxVar + uzVar) - 414.0*uxVar*uzVar
                -690.0 * (uxVar*uxVar + uzVar*uzVar) - 69.0 * uyVar*uyVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 

    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + pizz_I - 6.0*piyy_I + 34.0*pixz_I) - (2.0/69.0)*(-8.0-15.0*uxVar+8.0*uzVar)); //A30
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + pixx_I - 6.0*piyy_I + 34.0*pixz_I) - (2.0/69.0)*(-8.0+15.0*uzVar-8.0*uxVar));//A30  
    piyy = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 *pixx_I - 9.0*pizz_I + 54.0 * piyy_I - 30.0 * pixz_I) - (4.0/69.0) * (1.0- uxVar + uzVar)); //A31
    pixy = 9.0 * (inv_rho * 2.0 * pixy_I + ONETHIRD * uyVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I - ONETHIRD * uyVar); //A33
    pixz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * (-(-17.0*pixx_I - 17.0*pizz_I + 10.0*piyy_I) + 118.0*pixz_I) - (19.0/69.0)*(-1.0 - uxVar + uzVar)); //A32

    rhoVar = rho;             
                

}

//l1 = -1 // l2 = -1 // l3 = 0
//u1 = ux // u2  = uz // u3 == uywwd
__device__ void 
gpuBCMomentWB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0;  

    //IO: 1/5/7/9/11/13/15/16/18
    dfloat rho_I = pop[0] + pop[2] + pop[3] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12] + pop[14] + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[10] + pop[14] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[ 8]) - (pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[10]));
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[8] + pop[12] + pop[14] + pop[17] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12])-(pop[17]));
    dfloat pizz_I = inv_rho_I *  (pop[6] + pop[10] + pop[12] + pop[17] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * pizz_I 
                - 2.0*piyy_I - 19.0*pixz_I); //A28
    dfloat dE = 720.0 - 660.0 * (-uxVar - uzVar) + OMEGA * (430.0 + 30.0 * (uxVar + uzVar) + 414.0*uxVar*uzVar
                -690.0 * (uxVar*uxVar + uzVar*uzVar) - 69.0 * uyVar*uyVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;

    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + pizz_I - 6.0*piyy_I - 34.0*pixz_I) - (2.0/69.0)*(-8.0-15.0*uxVar-8.0*uzVar)); //A30
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + pixx_I - 6.0*piyy_I - 34.0*pixz_I) - (2.0/69.0)*(-8.0-15.0*uzVar-8.0*uxVar)); //A30
    piyy = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 *pixx_I - 9.0*pizz_I + 54.0 * piyy_I + 30.0 * pixz_I) - (4.0/69.0) * (1.0- uxVar - uzVar)); //A31
    pixy = 9.0 * (inv_rho * 2.0 * pixy_I + ONETHIRD * uyVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I + ONETHIRD * uyVar); //A33
    pixz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*pixx_I - 17.0*pizz_I + 10.0*piyy_I) + 118.0*pixz_I) - (19.0/69.0)*(1.0 - uxVar - uzVar)); //A32

    rhoVar = rho;
}

//l1 = 1 // l2 = 1 // l3 = 0
//u1 = ux // u2  = uz // u3 == uy
__device__ void 
gpuBCMomentEF(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){
    
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO:2/6/8/10/12/14/15/16/17
    dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[5] + pop[7] + pop[9] + pop[11]  + pop[13] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[9] + pop[13] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7]) - (pop[13]));
    dfloat pixz_I = inv_rho_I * ((pop[9])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[4] + pop[7] + pop[11] + pop[13] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11])-(pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[9] + pop[11] + pop[18] - cs2*rho_I);  
        
    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * pizz_I 
                - 2.0*piyy_I - 19.0*pixz_I); //A28  
    dfloat dE = 720.0 - 660.0 * (uxVar + uzVar) + OMEGA * (430.0 + 30.0 * (uxVar + uzVar) + 414.0*uxVar*uzVar
                -690.0 * (uxVar*uxVar + uzVar*uzVar) - 69.0 * uyVar*uyVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 
    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + pizz_I - 6.0*piyy_I - 34.0*pixz_I) - (2.0/69.0)*(-8.0+15.0*uxVar+8.0*uzVar)); //A30
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + pixx_I - 6.0*piyy_I - 34.0*pixz_I) - (2.0/69.0)*(-8.0+15.0*uzVar+8.0*uxVar)); //)A30    
    piyy = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 *pixx_I - 9.0*pizz_I + 54.0 * piyy_I + 30.0 * pixz_I) - (4.0/69.0) * (1.0+ uxVar + uzVar)); //A31 
    pixy = 9.0 * (inv_rho * 2.0 * pixy_I - ONETHIRD * uyVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I - ONETHIRD * uyVar); //A33 
    pixz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*pixx_I - 17.0*pizz_I + 10.0*piyy_I) + 118.0*pixz_I) - (19.0/69.0)*(1.0 + uxVar + uzVar)); //A32

    rhoVar = rho;                            

                  
}

//l1 = 1 // l2 = -1 // l3 = 0
//u1 = ux // u2  = uz // u3 == uy
__device__ void 
gpuBCMomentEB(dfloat *pop, dfloat &rhoVar, char dNodeType,
              dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
              dfloat &pixx, dfloat &pixy, dfloat &pixz,
              dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 
    
    //IO:2/5/8/9/10/11/14/16/18
    dfloat rho_I = pop[0] + pop[1] + pop[3] + pop[4] + pop[6] + pop[7] + pop[12] + pop[13]  + pop[15]  + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[13] + pop[15] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7]) - (pop[13]));
    dfloat pixz_I = inv_rho_I * (- (pop[15])) ;
    dfloat piyy_I = inv_rho_I * (pop[3] + pop[4] + pop[7] + pop[12] + pop[13] + pop[17] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12])-(pop[17]));
    dfloat pizz_I = inv_rho_I * (pop[6] + pop[12] + pop[15] + pop[17] - cs2*rho_I);

    dfloat bE = 1656.0 + 216.0 * T_OMEGA * (8.0 * pixx_I + 8.0 * pizz_I 
                - 2.0*piyy_I + 19.0*pixz_I); //A28
    dfloat dE = 720.0 - 660.0 * (uxVar - uzVar) + OMEGA * (430.0 + 30.0 * (uxVar + uzVar) - 414.0*uxVar*uzVar
                -690.0 * (uxVar*uxVar + uzVar*uzVar) - 69.0 * uyVar*uyVar); //A29

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;            

    //                                                                         v this sign may be wrong
    pixx = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pixx_I + pizz_I - 6.0*piyy_I + 34.0*pixz_I) - (2.0/69.0)*(-8.0+15.0*uxVar-8.0*uzVar)); //A30
    pizz = 4.5 * (inv_rho * (1.0/23.0) * rho_I * (47.0 * pizz_I + pixx_I - 6.0*piyy_I + 34.0*pixz_I) - (2.0/69.0)*(-8.0-15.0*uzVar+8.0*uxVar)); //A30
    piyy = 4.5 * (inv_rho * (2.0/69.0) * rho_I * (-9.0 *pixx_I - 9.0*pizz_I + 54.0 * piyy_I - 30.0 * pixz_I) - (4.0/69.0) * (1.0+ uxVar - uzVar)); //A31 
    pixy = 9.0 * (inv_rho * 2.0 * pixy_I - ONETHIRD * uyVar); //A33
    piyz = 9.0 * (inv_rho * 2.0 * piyz_I + ONETHIRD * uyVar); //A33   
    pixz = 9.0 * (inv_rho * (1.0/23.0) * rho_I * ((-17.0*pixx_I - 17.0*pizz_I + 10.0*piyy_I) + 118.0*pixz_I) - (19.0/69.0)*(1.0 + uxVar - uzVar)); //A32

    rhoVar = rho;     
}






//CORNERS

// l1 = -1
// l2 = +1
// l3 = +1
__device__ void 
gpuBCMomentNWF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/2/3/5/11/14/16
    dfloat rho_I = pop[0] + pop[2] + pop[3]  + pop[5] + pop[11] + pop[14] + pop[16];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[14] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * (- (pop[14]));
    dfloat pixz_I = inv_rho_I * (- (pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[11] + pop[14] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[11] + pop[16] - cs2*rho_I);


    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              +2.0 * pixy_I + 2.0 * pixz_I - 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(-uxVar + uyVar + uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( -uxVar * uyVar + -uxVar * uzVar + uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;        

    pixx = 4.5 *( inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(pixy_I + pixz_I + piyz_I)) + (2.0/9.0) * (1.0 + 2.0*uxVar + uyVar + uzVar));
    piyy = 4.5 *( inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixy_I - piyz_I - pixz_I)) + (2.0/9.0) * (1.0 - 2.0*uyVar - uxVar + uzVar));
    pizz = 4.5 *( inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixz_I - piyz_I - pixy_I)) + (2.0/9.0) * (1.0 - 2.0*uzVar - uxVar + uyVar));
    pixy = 9.0 *( inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * piyy_I - 3.0* pizz_I + 17.0*pixy_I - pixz_I + piyz_I) - (2.0/9.0) * (-1.0 - uyVar + uxVar - uzVar));
    pixz = 9.0 *( inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * pizz_I - 3.0* piyy_I + 17.0*pixz_I - pixy_I + piyz_I) - (2.0/9.0) * (-1.0 - uzVar + uxVar - uyVar));
    piyz = 9.0 *( inv_rho * ONETHIRD * (-3.0 * piyy_I -3.0 * pizz_I + 3.0* pixx_I + 17.0*piyz_I + pixy_I + pixz_I) - (2.0/9.0) * (1.0 + uzVar + uyVar - uxVar));

    rhoVar = rho;             
     
}

// l1 = -1
// l2 = +1
// l3 = -1
__device__ void 
gpuBCMomentNWB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){
                       
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/2/3/6/10/14/17
    dfloat rho_I = pop[0]  + pop[2] + pop[3] + pop[6] + pop[10] + pop[14] + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[10] + pop[14] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ( - (pop[14]));
    dfloat pixz_I = inv_rho_I * ((pop[9])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[14] + pop[17] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ( - (pop[17]));
    dfloat pizz_I = inv_rho_I *  (pop[6] + pop[10] + pop[17] - cs2*rho_I);


    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              +2.0 * pixy_I - 2.0 * pixz_I + 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(-uxVar + uyVar + -uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( -uxVar * uyVar + uxVar * uzVar - uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;         
       
    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(pixy_I - pixz_I - piyz_I)) + (2.0/9.0) * (1.0 + 2.0*uxVar + uyVar - uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixy_I + piyz_I + pixz_I)) + (2.0/9.0) * (1.0 - 2.0*uyVar - uxVar - uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixz_I + piyz_I - pixy_I)) + (2.0/9.0) * (1.0 + 2.0*uzVar - uxVar + uyVar));
    pixy = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * piyy_I - 3.0* pizz_I + 17.0*pixy_I + pixz_I - piyz_I) - (2.0/9.0) * (-1.0 - uyVar + uxVar + uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I -3.0 * pizz_I - 3.0* piyy_I + 17.0*pixz_I + pixy_I + piyz_I) - (2.0/9.0) * (1.0 - uzVar - uxVar + uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * piyy_I +3.0 * pizz_I - 3.0* pixx_I + 17.0*piyz_I - pixy_I + pixz_I) - (2.0/9.0) * (-1.0 + uzVar - uyVar + uxVar));

    rhoVar = rho;    
}

// l1 = +1
// l2 = +1
// l3 = +1
__device__ void 
gpuBCMomentNEF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){

    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/1/3/5/7/9/11
    dfloat rho_I = pop[0] + pop[1]  + pop[3]  + pop[5]  + pop[7]  + pop[9]  + pop[11];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[9] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7]));
    dfloat pixz_I = inv_rho_I * ((pop[9])) ;
    dfloat piyy_I = inv_rho_I *  (pop[3]  + pop[7]  + pop[11]  - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[11] ) );
    dfloat pizz_I = inv_rho_I *  (pop[5]  + pop[9]  + pop[11] - cs2*rho_I);


    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              -2.0 * pixy_I - 2.0 * pixz_I - 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(uxVar + uyVar + uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( uxVar * uyVar + uxVar * uzVar + uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;            

    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(-pixy_I - pixz_I + piyz_I)) + (2.0/9.0) * (1.0 - 2.0*uxVar + uyVar + uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixy_I - piyz_I + pixz_I)) + (2.0/9.0) * (1.0 - 2.0*uyVar + uxVar + uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixz_I - piyz_I + pixy_I)) + (2.0/9.0) * (1.0 - 2.0*uzVar + uxVar + uyVar));
    pixy = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * piyy_I + 3.0* pizz_I + 17.0*pixy_I - pixz_I - piyz_I) - (2.0/9.0) * (1.0 + uyVar + uxVar + uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * pizz_I + 3.0* piyy_I + 17.0*pixz_I - pixy_I - piyz_I) - (2.0/9.0) * (1.0 + uzVar + uxVar + uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (-3.0 * piyy_I -3.0 * pizz_I + 3.0* pixx_I + 17.0*piyz_I - pixy_I - pixz_I) - (2.0/9.0) * (1.0 + uzVar + uyVar + uxVar));

    rhoVar = rho;     
}


// l1 = +1
// l2 = +1
// l3 = -1
__device__ void 
gpuBCMomentNEB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/1/3/6/7/15/17
    dfloat rho_I = pop[0] + pop[1] + pop[3]  + pop[6] + pop[7] + pop[15]  + pop[17];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[7] + pop[15]  - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[7] ) );
    dfloat pixz_I = inv_rho_I * ( - (pop[15] )) ;
    dfloat piyy_I = inv_rho_I *  (pop[3] + pop[7] + pop[17]  - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ( - (pop[17] ));
    dfloat pizz_I = inv_rho_I *  (pop[6] + pop[15] + pop[17] - cs2*rho_I);                   


    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              -2.0 * pixy_I - 2.0 * pixz_I + 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(uxVar + uyVar -uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( uxVar * uyVar - uxVar * uzVar - uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;    

    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(-pixy_I + pixz_I - piyz_I)) + (2.0/9.0) * (1.0 - 2.0*uxVar + uyVar - uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixy_I + piyz_I - pixz_I)) + (2.0/9.0) * (1.0 - 2.0*uyVar + uxVar - uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixz_I + piyz_I + pixy_I)) + (2.0/9.0) * (1.0 + 2.0*uzVar + uxVar + uyVar));  
    pixy = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * piyy_I + 3.0* pizz_I + 17.0*pixy_I + pixz_I + piyz_I) - (2.0/9.0) * (1.0 + uyVar + uxVar - uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * pizz_I - 3.0* piyy_I + 17.0*pixz_I + pixy_I - piyz_I) - (2.0/9.0) * (-1.0 + uzVar - uxVar - uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * piyy_I +3.0 * pizz_I - 3.0* pixx_I + 17.0*piyz_I + pixy_I - pixz_I) - (2.0/9.0) * (-1.0 + uzVar - uyVar - uxVar));

    rhoVar = rho;      
}

// l1 = -1
// l2 = -1
// l3 = +1
__device__ void 
gpuBCMomentSWF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/2/4/5/8/16/18
    dfloat rho_I = pop[0]  + pop[2] + pop[4] + pop[5] + pop[8] + pop[16]  + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[16] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * (( pop[8]) );
    dfloat pixz_I = inv_rho_I * ( - ( pop[16])) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[8] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * (- (pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[16] + pop[18] - cs2*rho_I);


    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
                +2.0 * pixy_I + 2.0 * pixz_I - 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(-uxVar - uyVar + uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( uxVar * uyVar - uxVar * uzVar - uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;    

    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(-pixy_I + pixz_I - piyz_I)) + (2.0/9.0) * (1.0 + 2.0*uxVar - uyVar + uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixy_I + piyz_I - pixz_I)) + (2.0/9.0) * (1.0 + 2.0*uyVar - uxVar + uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixz_I + piyz_I + pixy_I)) + (2.0/9.0) * (1.0 - 2.0*uzVar - uxVar - uyVar));
    pixy = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * piyy_I + 3.0* pizz_I + 17.0*pixy_I + pixz_I + piyz_I) - (2.0/9.0) * (1.0 - uyVar - uxVar + uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * pizz_I - 3.0* piyy_I + 17.0*pixz_I + pixy_I - piyz_I) - (2.0/9.0) * (-1.0 - uzVar + uxVar + uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * piyy_I +3.0 * pizz_I - 3.0* pixx_I + 17.0*piyz_I + pixy_I - pixz_I) - (2.0/9.0) * (-1.0 - uzVar - uyVar + uxVar));

    rhoVar = rho;    

}

// l1 = -1
// l2 = -1
// l3 = -1
__device__ void 
gpuBCMomentSWB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/2/4/6/8/10/12
    dfloat rho_I = pop[0]  + pop[2] + pop[4] + pop[6] + pop[8] + pop[10] + pop[12];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[2] + pop[8] + pop[10] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ((pop[8]) );
    dfloat pixz_I = inv_rho_I * ((pop[10]) ) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[8] + pop[12] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12]) );
    dfloat pizz_I = inv_rho_I *  (pop[6] + pop[10] + pop[12] - cs2*rho_I);
    

    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              +2.0 * pixy_I + 2.0 * pixz_I + 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(-uxVar - uyVar - uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( uxVar * uyVar + uxVar * uzVar + uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho;

    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(-pixy_I - pixz_I + piyz_I)) + (2.0/9.0) * (1.0 + 2.0*uxVar - uyVar - uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixy_I - piyz_I + pixz_I)) + (2.0/9.0) * (1.0 + 2.0*uyVar - uxVar - uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixz_I - piyz_I + pixy_I)) + (2.0/9.0) * (1.0 + 2.0*uzVar - uxVar - uyVar)); 
    pixy = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * piyy_I + 3.0* pizz_I + 17.0*pixy_I - pixz_I - piyz_I) - (2.0/9.0) * (1.0 - uyVar - uxVar - uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * pizz_I + 3.0* piyy_I + 17.0*pixz_I - pixy_I - piyz_I) - (2.0/9.0) * (1.0 - uzVar - uxVar - uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (-3.0 * piyy_I -3.0 * pizz_I + 3.0* pixx_I + 17.0*piyz_I - pixy_I - pixz_I) - (2.0/9.0) * (1.0 - uzVar - uyVar - uxVar));

    rhoVar = rho;                     
}


// l1 = +1
// l2 = -1
// l3 = +1
__device__ void 
gpuBCMomentSEF(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){
    
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/1/4/5/9/13/18
    dfloat rho_I = pop[0] + pop[1] + pop[4] + pop[5] + pop[9] + pop[13] + pop[18];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1] + pop[9] + pop[13]  - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ( - (pop[13]));
    dfloat pixz_I = inv_rho_I * ((pop[9]) ) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[13] + pop[18] - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ( - (pop[18]));
    dfloat pizz_I = inv_rho_I *  (pop[5] + pop[9] + pop[18] - cs2*rho_I);


    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              -2.0 * pixy_I + 2.0 * pixz_I - 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(uxVar - uyVar + uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( -uxVar * uyVar + uxVar * uzVar - uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 

    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(pixy_I - pixz_I - piyz_I)) + (2.0/9.0) * (1.0 - 2.0*uxVar - uyVar + uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixy_I + piyz_I + pixz_I)) + (2.0/9.0) * (1.0 + 2.0*uyVar + uxVar + uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(-pixz_I + piyz_I - pixy_I)) + (2.0/9.0) * (1.0 - 2.0*uzVar + uxVar - uyVar));
    pixy = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * piyy_I - 3.0* pizz_I + 17.0*pixy_I - pixz_I - piyz_I) - (2.0/9.0) * (-1.0 + uyVar - uxVar + uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (-3.0 * pixx_I -3.0 * pizz_I + 3.0* piyy_I + 17.0*pixz_I + pixy_I + piyz_I) - (2.0/9.0) * (1.0 + uzVar + uxVar + uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * piyy_I +3.0 * pizz_I - 3.0* pixx_I + 17.0*piyz_I - pixy_I + pixz_I) - (2.0/9.0) * (-1.0 - uzVar + uyVar + uxVar));

    rhoVar = rho;                     
}

// l1 = +1
// l2 = -1
// l3 = -1
__device__ void 
gpuBCMomentSEB(dfloat *pop, dfloat &rhoVar, char dNodeType,
               dfloat &uxVar, dfloat &uyVar, dfloat &uzVar,
               dfloat &pixx, dfloat &pixy, dfloat &pixz,
               dfloat &piyy, dfloat &piyz, dfloat &pizz){
    
    uxVar = 0.0;  
    uyVar = 0.0;  
    uzVar = 0.0; 

    //IO: 0/1/4/6/12/13/15
    dfloat rho_I = pop[0] + pop[1]  + pop[4] + pop[6] + pop[12] + pop[13] + pop[15];
    dfloat inv_rho_I = 1.0 / rho_I;
    dfloat pixx_I = inv_rho_I *  (pop[1]  + pop[13] + pop[15] - cs2*rho_I);
    dfloat pixy_I = inv_rho_I * ( - (pop[13]));
    dfloat pixz_I = inv_rho_I * ( - (pop[15])) ;
    dfloat piyy_I = inv_rho_I *  (pop[4] + pop[12] + pop[13]  - cs2*rho_I);
    dfloat piyz_I = inv_rho_I * ((pop[12]));
    dfloat pizz_I = inv_rho_I *  ( pop[6] + pop[12] + pop[15] - cs2*rho_I);
    

    dfloat bE = (1.0/24.0)*(T_OMEGA)*(pixx_I + piyy_I + pizz_I 
              -2.0 * pixy_I + 2.0 * pixz_I + 2.0 * piyz_I);
    dfloat dE = 4.0 + 10.0*OMEGA + 4*(OMEGA-3.0)*(uxVar - uyVar - uzVar) 
              - 9.0*OMEGA*(uxVar*uxVar + uyVar*uyVar + uzVar*uzVar) + 6.0* OMEGA *
              ( -uxVar * uyVar - uxVar * uzVar + uyVar*uzVar);

    dfloat rho = rho_I * bE / dE; //A27
    dfloat inv_rho = 1.0/rho; 

    pixx = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pixx_I - 2.0*piyy_I - 2.0 * pizz_I + 6.0*(pixy_I + pixz_I + piyz_I)) + (2.0/9.0) * (1.0 - 2.0*uxVar - uyVar - uzVar));
    piyy = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * piyy_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixy_I - piyz_I - pixz_I)) + (2.0/9.0) * (1.0 + 2.0*uyVar + uxVar - uzVar));
    pizz = 4.5 * (inv_rho * ONETHIRD * rho_I * (10.0 * pizz_I - 2.0*pixx_I - 2.0 * pizz_I + 6.0*(pixz_I - piyz_I - pixy_I)) + (2.0/9.0) * (1.0 + 2.0*uzVar + uxVar - uyVar)); 
    pixy = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * piyy_I - 3.0* pizz_I + 17.0*pixy_I - pixz_I + piyz_I) - (2.0/9.0) * (-1.0 + uyVar - uxVar + uzVar));
    pixz = 9.0 * (inv_rho * ONETHIRD * (+3.0 * pixx_I +3.0 * pizz_I - 3.0* piyy_I + 17.0*pixz_I - pixy_I + piyz_I) - (2.0/9.0) * (-1.0 + uzVar - uxVar + uyVar));
    piyz = 9.0 * (inv_rho * ONETHIRD * (-3.0 * piyy_I -3.0 * pizz_I + 3.0* pixx_I + 17.0*piyz_I + pixy_I + pixz_I) - (2.0/9.0) * (1.0 - uzVar - uyVar + uxVar));

    rhoVar = rho;    
}