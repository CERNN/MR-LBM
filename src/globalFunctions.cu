#include "globalFunctions.h"

__host__ __device__
dfloat dot_product(dfloat3 v1, dfloat3 v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__
dfloat3 cross_product(dfloat3 v1, dfloat3 v2) {
    dfloat3 cross;
    cross.x = v1.y * v2.z - v1.z * v2.y;
    cross.y = v1.z * v2.x - v1.x * v2.z;
    cross.z = v1.x * v2.y - v1.y * v2.x;
    return cross;
}

__host__ __device__
dfloat vector_length(dfloat3 v) {
    return  sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__
dfloat3 vector_normalize(dfloat3 v) {
    dfloat inv_length =  rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    dfloat3 norm_v;
    if (isnan(inv_length)||isinf(inv_length)){
        norm_v.x = 0.0;
        norm_v.y = 0.0;
        norm_v.z = 0.0;
    }else{
        norm_v.x = v.x * inv_length;
        norm_v.y = v.y * inv_length;
        norm_v.z = v.z * inv_length;
    }
    return norm_v;
}

__host__ __device__
void transpose_matrix_3x3(dfloat matrix[3][3], dfloat result[3][3]) {
    result[0][0] = matrix[0][0];   result[0][1] = matrix[1][0];   result[0][2] = matrix[2][0];
    result[1][0] = matrix[0][1];   result[1][1] = matrix[1][1];   result[1][2] = matrix[2][1];
    result[2][0] = matrix[0][2];   result[2][1] = matrix[1][2];   result[2][2] = matrix[2][2];
}

__host__ __device__
void multiply_matrices_3x3(dfloat A[3][3], dfloat B[3][3], dfloat result[3][3]) {
    result[0][0] = A[0][0] * B[0][0] + A[1][0] * B[0][1] + A[2][0] * B[0][2];
    result[1][0] = A[0][0] * B[1][0] + A[1][0] * B[1][1] + A[2][0] * B[1][2];
    result[2][0] = A[0][0] * B[2][0] + A[1][0] * B[2][1] + A[2][0] * B[2][2];

    result[0][1] = A[0][1] * B[0][0] + A[1][1] * B[0][1] + A[2][1] * B[0][2];
    result[1][1] = A[0][1] * B[1][0] + A[1][1] * B[1][1] + A[2][1] * B[1][2];
    result[2][1] = A[0][1] * B[2][0] + A[1][1] * B[2][1] + A[2][1] * B[2][2];

    result[0][2] = A[0][2] * B[0][0] + A[1][2] * B[0][1] + A[2][2] * B[0][2];
    result[1][2] = A[0][2] * B[1][0] + A[1][2] * B[1][1] + A[2][2] * B[1][2];
    result[2][2] = A[0][2] * B[2][0] + A[1][2] * B[2][1] + A[2][2] * B[2][2];
}

__host__ __device__
void add_matrices_3x3(dfloat scalar, dfloat A[3][3], dfloat B[3][3], dfloat result[3][3]) {
    result[0][0] = scalar * A[0][0] + B[0][0];
    result[0][1] = scalar * A[0][1] + B[0][1];
    result[0][2] = scalar * A[0][2] + B[0][2];
    
    result[1][0] = scalar * A[1][0] + B[1][0];
    result[1][1] = scalar * A[1][1] + B[1][1];
    result[1][2] = scalar * A[1][2] + B[1][2];
    
    result[2][0] = scalar * A[2][0] + B[2][0];
    result[2][1] = scalar * A[2][1] + B[2][1];
    result[2][2] = scalar * A[2][2] + B[2][2];
}

__host__ __device__
dfloat determinant_3x3(dfloat A[3][3]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) 
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) 
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

__host__ __device__
void adjugate_3x3(dfloat A[3][3], dfloat adj[3][3]) {
    adj[0][0] = A[1][1] * A[2][2] - A[1][2] * A[2][1];
    adj[0][1] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]);
    adj[0][2] = A[1][0] * A[2][1] - A[1][1] * A[2][0];
    
    adj[1][0] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]);
    adj[1][1] = A[0][0] * A[2][2] - A[0][2] * A[2][0];
    adj[1][2] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]);
    
    adj[2][0] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
    adj[2][1] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]);
    adj[2][2] = A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

__host__ __device__
void inverse_3x3(dfloat A[3][3], dfloat result[3][3]) {
    dfloat det = determinant_3x3(A);
        
    dfloat adj[3][3];
    adjugate_3x3(A, adj);
    
    dfloat inv_det = 1.0 / det;
    
    // Compute the inverse matrix without loops
    result[0][0] = adj[0][0] * inv_det;
    result[0][1] = adj[0][1] * inv_det;
    result[0][2] = adj[0][2] * inv_det;
    
    result[1][0] = adj[1][0] * inv_det;
    result[1][1] = adj[1][1] * inv_det;
    result[1][2] = adj[1][2] * inv_det;
    
    result[2][0] = adj[2][0] * inv_det;
    result[2][1] = adj[2][1] * inv_det;
    result[2][2] = adj[2][2] * inv_det;
}

__host__ __device__
void dfloat6_to_matrix(dfloat6 I, dfloat M[3][3]) {
    M[0][0] = I.xx;     M[1][0] = I.xy;     M[2][0] = I.xz; 
    M[0][1] = I.xy;     M[1][1] = I.yy;     M[2][1] = I.yz;
    M[0][2] = I.xz;     M[1][2] = I.yz;     M[2][2] = I.zz;

}

__host__ __device__
dfloat6 matrix_to_dfloat6(dfloat M[3][3]) {
    dfloat6 I;
    I.xx = M[0][0];    I.xy = M[1][0];    I.xz = M[2][0];    
                       I.yy = M[1][1];    I.yz = M[2][1];    
                                          I.zz = M[2][2];

    return I;
}

__host__ __device__
dfloat4 quart_conjugate(dfloat4 q) {
    dfloat4 q_conj;
    q_conj.w = q.w;
    q_conj.x = -q.x;
    q_conj.y = -q.y;
    q_conj.z = -q.z;
    return q_conj;
}

__host__ __device__
void quart_to_rotation_matrix(dfloat4 q, dfloat R[3][3]){
    dfloat qx2 = q.x * q.x;
    dfloat qy2 = q.y * q.y;
    dfloat qz2 = q.z * q.z;
    dfloat qwqx = q.w * q.x;
    dfloat qwqy = q.w * q.y;
    dfloat qwqz = q.w * q.z;
    dfloat qxqy = q.x * q.y;
    dfloat qxqz = q.x * q.z;
    dfloat qyqz = q.y * q.z;

    R[0][0] = 1 - 2 * (qy2 + qz2);  R[1][0] = 2 * (qxqy - qwqz);    R[2][0] = 2 * (qxqz + qwqy);
    R[0][1] = 2 * (qxqy + qwqz);    R[1][1] = 1 - 2 * (qx2 + qz2);  R[2][1] = 2 * (qyqz - qwqx);
    R[0][2] = 2 * (qxqz - qwqy);    R[1][2] = 2 * (qyqz + qwqx);    R[2][2] = 1 - 2 * (qx2 + qy2);
}

__host__ __device__
dfloat3 rotate_vector_by_matrix(dfloat R[3][3],dfloat3 v) {
    dfloat3 v_rot;
    v_rot.x = R[0][0] * v.x + R[1][0] * v.y + R[2][0] * v.z;
    v_rot.y = R[0][1] * v.x + R[1][1] * v.y + R[2][1] * v.z;
    v_rot.z = R[0][2] * v.x + R[1][2] * v.y + R[2][2] * v.z;
    return v_rot;
}

__host__ __device__
dfloat3 rotate_vector_by_quart_R(dfloat3 v, dfloat4 q){
    dfloat R[3][3];
    quart_to_rotation_matrix(q, R);
    return rotate_vector_by_matrix(R,v);
}

__host__ __device__
dfloat4 compute_rotation_quart(dfloat3 v1, dfloat3 v2) {
    v1 = vector_normalize(v1);
    v2 = vector_normalize(v2);

    dfloat dot = dot_product(v1, v2);

    // Calculate the angle of rotation
    dfloat angle = acos(dot);

    // Calculate the axis of rotation
    dfloat3 axis = cross_product(v1, v2);
    axis = vector_normalize(axis);

    dfloat4 q;
    q.w = cos(angle / 2.0);
    q.x = axis.x * sin(angle / 2.0);
    q.y = axis.y * sin(angle / 2.0);
    q.z = axis.z * sin(angle / 2.0);

    return q;
}

__host__ __device__
dfloat4 axis_angle_to_quart(dfloat3 axis, dfloat angle) {
    dfloat4 q;
    
    // Normalize the axis of rotation
    axis = vector_normalize(axis);
    
    // Compute the quaternion
    q.w = cos(angle / 2.0);
    q.x = axis.x * sin(angle / 2.0);
    q.y = axis.y * sin(angle / 2.0);
    q.z = axis.z * sin(angle / 2.0);
    
    return q;
}

__host__ __device__
void rotate_matrix_by_R_w_quart(dfloat4 q, dfloat I[3][3]) {

    dfloat R[3][3];
    dfloat Rt[3][3];
    dfloat temp[3][3];

    //compute rotation matrix
    quart_to_rotation_matrix(q,R);  
    //compute tranposte
    transpose_matrix_3x3(R,Rt);
    //perform rotation R*I*R^t
    multiply_matrices_3x3(R,I,temp);
    multiply_matrices_3x3(temp,Rt,I);
}

__host__ __device__
dfloat6 rotate_inertia_by_quart(dfloat4 q, dfloat6 I6) {
    dfloat I[3][3];

    dfloat6_to_matrix(I6,I);
    rotate_matrix_by_R_w_quart(q,I);
    I6 = matrix_to_dfloat6(I);  
    return I6;

}