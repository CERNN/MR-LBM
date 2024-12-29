// TODO NOT WORKING. NEED PROPER TREATMENT OF THE GRADIENT OF THE CONFORMATION TENSOR


dfloat aa, bb;
#ifdef FENE_P
    //trace of A
    dfloat trA = AxxVar + AyyVar + AzzVar;
    aa = -inv_fenep_lambda/(1-trA/(fenep_re*fenep_re));
    bb = -inv_fenep_lambda/(1-3.0/(fenep_re*fenep_re));
    #ifdef OLDROYD_B
    aa = 1.0;
    bb = 1.0;
    #endif
#endif

dfloat switch_conf = 0.0;
if (step>12500){
    switch_conf = 1.0;     
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//create Matrix of conformation based on PSI

dfloat A[3][3]= { //psi matrix
    {AxxVar, AxyVar, AxzVar},
    {AxyVar, AyyVar, AyzVar},
    {AxzVar, AyzVar, AzzVar}    
};

// Compute coefficients of the characteristic polynomial
dfloat a = -1.0;
dfloat b = A[0][0] + A[1][1] + A[2][2];
dfloat c = A[0][0] * A[1][1] + A[0][0] * A[2][2] + A[1][1] * A[2][2] - A[0][1] * A[1][0] - A[0][2] * A[2][0] - A[1][2] * A[2][1];
dfloat d = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

// Solve for eigenvalues
dfloat p = (3.0 * a * c - b * b) / (3.0 * a * a);
dfloat q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);


dfloat discriminant = (q * q / 4.0) + (p * p * p / 27.0);
dfloat eigenvalues[3];

if (discriminant > FLT_MIN) {
    // One real root
    dfloat u = cbrt(-q / 2.0 + sqrt(discriminant));
    dfloat v = cbrt(-q / 2.0 - sqrt(discriminant));
    eigenvalues[0] = u + v - b / (3.0 * a);
} else if (fabs(discriminant) < FLT_MIN) {
    // Triple or double root
    dfloat u = cbrt(-q / 2.0);
    eigenvalues[0] = 2.0 * u - b / (3.0 * a);
    eigenvalues[1] = -u - b / (3.0 * a);
} else {
    // Three distinct real roots
    dfloat r = sqrt(-p * p * p / 27.0);
    dfloat theta = acos(-q / (2.0 * r));
    r = cbrt(r);
    eigenvalues[0] = 2.0 * r * cos(theta / 3.0) - b / (3.0 * a);
    eigenvalues[1] = 2.0 * r * cos((theta + 2.0 * M_PI) / 3.0) - b / (3.0 * a);
    eigenvalues[2] = 2.0 * r * cos((theta + 4.0 * M_PI) / 3.0) - b / (3.0 * a);
}

// Compute normalized eigenvectors 
dfloat R[3][3];
for (int k = 0; k < 3; k++) {
    dfloat lambda = eigenvalues[k];
    dfloat temp[3][3] = {
        {A[0][0] - lambda, A[0][1], A[0][2]},
        {A[1][0], A[1][1] - lambda, A[1][2]},
        {A[2][0], A[2][1], A[2][2] - lambda}
    };
    dfloat v[3] = {temp[1][1] * temp[2][2] - temp[1][2] * temp[2][1],
                    temp[1][2] * temp[2][0] - temp[1][0] * temp[2][2],
                    temp[1][0] * temp[2][1] - temp[1][1] * temp[2][0]};
    dfloat norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    R[0][k] = v[0] / norm;
    R[1][k] = v[1] / norm;
    R[2][k] = v[2] / norm;
}

//Transpose R into R^T
dfloat Rt[3][3];
transpose_matrix_3x3(R, Rt);

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//temporary matrices
dfloat temp[3][3];
dfloat temp2[3][3] = {0.0};
dfloat sum_temp[3][3] = {0.0};

//velocity gradient matrix
dfloat GU[3][3] = {
    {duxdx_t30/3.0, duydx_t30/3.0, duzdx_t30/3.0},
    {duxdy_t30/3.0, duydy_t30/3.0, duzdy_t30/3.0},
    {duxdz_t30/3.0, duydz_t30/3.0, duzdz_t30/3.0}    
};

// Compute M = R^T * GU * R //eq.15
dfloat M[3][3] = {0.0};
multiply_matrices_3x3(GU, Rt, temp); //GU*Rt
multiply_matrices_3x3(R, temp, M);   //R* (GU*Rt)


// Compute Omega matrix Eq.16a
dfloat OHM[3][3] = {0.0};
//i  j       j  j      i  j      i  i      j  i        i  i      j  j
OHM[0][1] = (A[1][1] * M[0][1] + A[0][0] * M[1][0]) / (A[0][0] - A[1][1]);
OHM[0][2] = (A[2][2] * M[0][2] + A[0][0] * M[2][0]) / (A[0][0] - A[2][2]);
OHM[1][0] = (A[0][0] * M[1][0] + A[1][1] * M[0][1]) / (A[1][1] - A[0][0]);
OHM[1][2] = (A[2][2] * M[1][2] + A[1][1] * M[2][1]) / (A[1][1] - A[2][2]);
OHM[2][0] = (A[0][0] * M[2][0] + A[2][2] * M[0][2]) / (A[2][2] - A[0][0]);
OHM[2][1] = (A[1][1] * M[2][1] + A[2][2] * M[1][2]) / (A[2][2] - A[1][1]);

multiply_matrices_3x3(R, OHM, temp); //R*O
multiply_matrices_3x3(temp, Rt, OHM); //(R*0) * R^T

// Compute B matrix Eq 16.b
dfloat B[3][3] = {0.0};
// the element are diogonal so its not hard to expand
B[0][0] = M[0][0] * R[0][0] * R[0][0] + M[1][1] * R[0][1] * R[0][1] + M[2][2] * R[0][2] * R[0][2];
B[0][1] = M[0][0] * R[0][0] * R[1][0] + M[1][1] * R[0][1] * R[1][1] + M[2][2] * R[0][2] * R[1][2];
B[0][2] = M[0][0] * R[0][0] * R[2][0] + M[1][1] * R[0][1] * R[2][1] + M[2][2] * R[0][2] * R[2][2];

B[1][0] = M[0][0] * R[1][0] * R[0][0] + M[1][1] * R[1][1] * R[0][1] + M[2][2] * R[1][2] * R[0][2];
B[1][1] = M[0][0] * R[1][0] * R[1][0] + M[1][1] * R[1][1] * R[1][1] + M[2][2] * R[1][2] * R[1][2];
B[1][2] = M[0][0] * R[1][0] * R[2][0] + M[1][1] * R[1][1] * R[2][1] + M[2][2] * R[1][2] * R[2][2];

B[2][0] = M[0][0] * R[2][0] * R[0][0] + M[1][1] * R[2][1] * R[0][1] + M[2][2] * R[2][2] * R[0][2];
B[2][1] = M[0][0] * R[2][0] * R[1][0] + M[1][1] * R[2][1] * R[1][1] + M[2][2] * R[2][2] * R[1][2];
B[2][2] = M[0][0] * R[2][0] * R[2][0] + M[1][1] * R[2][1] * R[2][1] + M[2][2] * R[2][2] * R[2][2];

/* alternatively
B[0][0] = M[0][0];
B[1][1] = M[1][1];
B[2][2] = M[2][2];

multiply_matrices_3x3(R, B, temp); //R*B
multiply_matrices_3x3(temp, Rt, B); //(R*B) * R^T
*/

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// need compute eq.17 now
// the force term becomes R*(1/lambda)*(A^-1 - I)*Rt + 2B + OHM*Psi - Psi*OHM
//                     R*(A^-1/lambda - I/lambda)*Rt + 2B + OHM*Psi - Psi*OHM

// -I/lambda
dfloat IM[3][3] = {0.0};
IM[0][0] = -inv_fenep_lambda*bb;
IM[1][1] = -inv_fenep_lambda*bb;
IM[2][2] = -inv_fenep_lambda*bb;


// A^-1
inverse_3x3(A,temp);

// temp2 = A/lambda - I/lambda
add_matrices_3x3(inv_fenep_lambda*aa,temp,IM,temp);
//temp = R*(1/lambda)*(A^-1-I)
multiply_matrices_3x3(R, temp, temp2); 
//sum_temp =  R*(1/lambda)*(A^-1-I)*R^T
multiply_matrices_3x3(temp2, Rt, sum_temp); 

//sum_temp =  R*(1/lambda)*(A^-1-I)*R^T + 2*B
add_matrices_3x3(2.0,B,sum_temp,sum_temp); 

//define PSI
dfloat PSI[3][3]= {
    {AxxVar, AxyVar, AxzVar},
    {AxyVar, AyyVar, AyzVar},
    {AxzVar, AyzVar, AzzVar}    
};

//sum_temp =  R*(1/lambda)*(A^-1-I)*R^T + 2*B + OHM * Psi
multiply_matrices_3x3(OHM, PSI, temp); 
add_matrices_3x3(1.0,temp,sum_temp,sum_temp);

//sum_temp =  R*(1/lambda)*(A^-1-I)*Rt + 2B + OHM*Psi - Psi*OHM
multiply_matrices_3x3(PSI, OHM, temp);
add_matrices_3x3(-1.0,temp,sum_temp,sum_temp); 


// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dfloat G_xx = sum_temp[0][0];
dfloat G_yy = sum_temp[1][1];
dfloat G_zz = sum_temp[2][2];

dfloat G_xy =  sum_temp[0][1];
dfloat G_xz =  sum_temp[0][2];
dfloat G_yz =  sum_temp[1][2];


fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XX_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_xx;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_xy;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_XZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0*G_xz;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YY_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = switch_conf*G_yy;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_YZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0*G_yz;
fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, G_ZZ_C_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0*G_zz;


// now compute the gradient of the conformation ???
// c = R * exp(eigen value) * Rt
dfloat exp_eigen_0 = exp(eigenvalues[0]);
dfloat exp_eigen_1 = exp(eigenvalues[1]);
dfloat exp_eigen_2 = exp(eigenvalues[2]);

dfloat C[3][3]
dfloat Cxx, Cxy, Cxz, Cyy, Cyz, Czz;

Cxx = exp_eigen_0 * R[0][0] * R[0][0] + exp_eigen_1 * R[0][1] * R[0][1] + exp_eigen_2 * R[0][2] * R[0][2];
Cxy = exp_eigen_0 * R[0][0] * R[1][0] + exp_eigen_1 * R[0][1] * R[1][1] + exp_eigen_2 * R[0][2] * R[1][2];
Cxz = exp_eigen_0 * R[0][0] * R[2][0] + exp_eigen_1 * R[0][1] * R[2][1] + exp_eigen_2 * R[0][2] * R[2][2];

Cyy = exp_eigen_0 * R[1][0] * R[1][0] + exp_eigen_1 * R[1][1] * R[1][1] + exp_eigen_2 * R[1][2] * R[1][2];
Cyz = exp_eigen_0 * R[1][0] * R[2][0] + exp_eigen_1 * R[1][1] * R[2][1] + exp_eigen_2 * R[1][2] * R[2][2];

Czz = exp_eigen_0 * R[2][0] * R[2][0] + exp_eigen_1 * R[2][1] * R[2][1] + exp_eigen_2 * R[2][2] * R[2][2];


#include "includeFiles/conformationTransport/log_conformation_gradient.inc"

//need compute the eigen values /vectors -> store those -> then exponential the values and multiply by the vectors.

L_Fx += (dCxxdx + dCxydy + dCxzdz)*(fenep_mu_d_lambda)*switch_conf; //zero just to debug for now
L_Fy += (dCxydx + dCyydy + dCyzdz)*(fenep_mu_d_lambda)*switch_conf; //zero just to debug for now
L_Fz += (dCxzdx + dCyzdy + dCzzdz)*(fenep_mu_d_lambda)*0.0; //zero just to debug for now





