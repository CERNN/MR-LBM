#ifndef __NNF_H
#define __NNF_H

#include <math.h>
#include <cmath>
#include "var.h"

/* -------------------------------------------------------------------------- */
/* ------------------------ NON NEWTONIAN FLUID TYPE ------------------------ */
/* -------------------------------------------------------------------------- */

/* ------------------------------- POWER- LAW ------------------------------- */

#ifdef POWERLAW
constexpr dfloat N_INDEX = 0.5;                         // Power index
constexpr dfloat K_CONSISTENCY = RHO_0*(TAU-0.5)/3;      // Consistency factor
constexpr dfloat GAMMA_0 = 0;       // Truncated Power-Law. 
                                    // Leave as 0 to no truncate
#define OMEGA_LAST_STEP // Needs omega from last step
#endif
/* --------------------------------BINGHAM---------------------------------- */
#ifdef BINGHAM
// Inputs
constexpr dfloat S_YY = BN_NUMBER *VISC*invSqrtt(L/(T_gravity_t_beta*T_DELTA_T));
constexpr dfloat S_Y= S_YY;
constexpr dfloat Bn_turan = S_Y*sqrtt(L/(T_gravity_t_beta*T_DELTA_T))/VISC;
constexpr dfloat Bn_Zhang = Bn_turan * sqrtt(T_PR_NUMBER/T_RA_NUMBER);
constexpr dfloat Y_number = S_Y /(RHO_0*T_gravity_t_beta*L*T_DELTA_T);

//constexpr dfloat Bn = 10;
//constexpr dfloat S_Y = Bn * VISC * U_MAX / L;                // Yield stress 0.00579
// Calculated variables
constexpr dfloat OMEGA_P = 1 / (3.0*VISC+0.5);    // 1/tau_p = 1/(3*eta_p+0.5)
#endif
/* ------------------------------KEE_TURCOTEE-------------------------------- */
#ifdef BI_VISCOSITY

constexpr dfloat Bn = 0.4;
constexpr dfloat S_Y = Bn*VISC*invSqrtt(2*L/(T_gravity_t_beta*T_DELTA_T));

constexpr dfloat VISC_RATIO = 1.0/1000.0; // MU/MU_Y
constexpr dfloat ETA_Y = VISC / VISC_RATIO ;
constexpr dfloat TAU_Y = 3*ETA_Y + 0.5;
constexpr dfloat OMEGA_Y = 1.0/TAU_Y;
constexpr dfloat OMEGA_P = 1 / (3.0*VISC+0.5);
constexpr dfloat GAMMA_C = S_Y / ETA_Y;

#endif
/* -------------------------------------------------------------------------- */

/* ------------------------------KEE_TURCOTEE-------------------------------- */
#ifdef KEE_TURCOTEE

constexpr dfloat S_Y = 0;
constexpr dfloat t1 = 1e-3;
constexpr dfloat eta_0 =  1e-3;

#endif
/* -------------------------------------------------------------------------- */




#ifdef OMEGA_FIELD
    #ifdef POWER_LAW
    __device__ 
    dfloat calcOmega(dfloat omegaOld, dfloat const auxStressMag, const int step){
        omega = omegaOld; //initial guess

        dfloat fx, fx_dx;
        const dfloat c_s_2 = 1.0/3.0;
        const dfloat a = K_CONSISTENCY*POW_FUNCTION(auxStressMag / (RHO_0 * c_s_2) ,N_INDEX);
        const dfloat b = 0.5 * auxStressMag;
        const dfloat c = -auxStressMag;

        if(auxStressMag < 1e-6)
            return omega = 0;

        //#pragma unroll
        for (int i = 0; i< 7;i++){
            fx = a * POW_FUNCTION (omega,N_INDEX) + b * omega + c;
            fx_dx = a * N_INDEX * POW_FUNCTION (omega,N_INDEX - 1.0) + b ;

            if (abs(fx/fx_dx) < 1e-6){
                break;
            } //convergence criteria
                
            omega = omega - fx / fx_dx;
        }

        return omega;
    }
    #endif 


    #ifdef BINGHAM
    __device__ 
    dfloat __forceinline__ calcOmega(dfloat omegaOld, dfloat const auxStressMag, const int step){
        return OMEGA_P * myMax(0.0, (1 - S_Y / auxStressMag));
        //return OMEGA_P * myMax(0.0, (1 - S_Y * ((dfloat)(step - NNF_TRIGGER_STEP)/(NNF_TRIGGER_STEP_SIZE)) / auxStressMag));
    }
    #endif  

    #ifdef BI_VISCOSITY
    __device__ 
    dfloat __forceinline__ calcOmega(dfloat omegaOld, dfloat const auxStressMag){
        return  myMax(OMEGA_Y, OMEGA_P *(1 - S_Y *(1-VISC_RATIO) / auxStressMag));
    }
    #endif  


    // NOT TESTE/VALIDATED https://arxiv.org/abs/2401.02942 has analythical solution
    #ifdef KEE_TURCOTEE
    __device__ 
    dfloat calcOmega(dfloat omegaOld, dfloat const auxStressMag, const int step){
        const dfloat A = auxStressMag/2;
        const dfloat B = auxStressMag/(RHO_0*cs2);
        const dfloat C = B*eta_0;
        const dfloat D = -t1*B;
        const dfloat E = S_Y - auxStressMag;

        if(auxStressMag < 1e-6)
            return omega = 0;
        //#pragma unroll
        for (int i = 0; i< 7;i++){
            fx = omega*(A+C*__expf(D*omega)) + E;
            fx_dx = A + C*__expf(D*omega)*(1+D*omega);

            if (abs(fx/fx_dx) < 1e-6){
                break;
            } //convergence criteria
                
            omega = omega - fx / fx_dx;
        }

        return omega;
    }
    #endif


#endif // OMEGA_FIELD


#endif // __NNF_H