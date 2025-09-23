#include "globalFunctions.h"


__host__ __device__
dfloat clamp01(dfloat value) {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

__host__ __device__
dfloat3 vector_lerp(dfloat3 v1, dfloat3 v2, dfloat t) {
    return dfloat3(v1.x + t * (v2.x - v1.x), v1.y + t * (v2.y - v1.y), v1.z + t * (v2.z - v1.z));
}


// ****************************************************************************
// ************************   VECTOR OPERATIONS   *****************************
// ****************************************************************************

__device__
dfloat3 planeProjection(dfloat3 P, dfloat3 n, dfloat d) {
    // Copy original coordinates
    dfloat3 proj = P;
    
    const dfloat EPSILON = 1e-6;
    // Update projection based on the direction of the normal vector
    if (fabs(n.x - 1.0) < EPSILON) {
        proj.x = 0.0;
    } else if (fabs(n.x + 1.0) < EPSILON) {
        proj.x = d;
    }

    if (fabs(n.y - 1.0) < EPSILON) {
        proj.y = 0.0;
    } else if (fabs(n.y + 1.0) < EPSILON) {
        proj.y = d;
    }

    if (fabs(n.z - 1.0) < EPSILON) {
        proj.z = 0.0;
    } else if (fabs(n.z + 1.0) < EPSILON) {
        proj.z = d;
    }

    return proj;
}

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

__device__
dfloat point_to_point_distance_periodic(dfloat3 p1, dfloat3 p2) {
    dfloat3 delta = p1 - p2;

    #ifdef BC_X_PERIODIC
    if (delta.x > NX / 2.0) delta.x -= NX;
    if (delta.x < -NX / 2.0) delta.x += NX;
    #endif //BC_X_PERIODIC
    #ifdef BC_Y_PERIODIC
    if (delta.y > NY / 2.0) delta.y -= NY;
    if (delta.y < -NY / 2.0) delta.y += NY;
    #endif //BC_Y_PERIODIC
    #ifdef BC_Z_PERIODIC
    if (delta.z > NZ / 2.0) delta.z -= NZ;
    if (delta.z < -NZ / 2.0) delta.z += NZ;
    #endif //BC_Z_PERIODIC

    return sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
}

// Helper function to compute vector difference with periodic/wall BCs
__device__ dfloat3 getDiffPeriodic(const dfloat3& p1, const dfloat3& p2) {
    dfloat dx, dy, dz;
    // X direction
    #ifdef BC_X_PERIODIC
    dx = abs(p1.x - p2.x) > ((NX-1) / 2.0) ?
        (p1.x < p2.x ? (p1.x + (NX-1) - p2.x) : (p1.x - (NX-1) - p2.x))
        : p1.x - p2.x;
    #else
    dx = p1.x - p2.x;
    #endif
    // Y direction
    #ifdef BC_Y_PERIODIC
    dy = abs(p1.y - p2.y) > ((NY-1) / 2.0) ?
        (p1.y < p2.y ? (p1.y + (NY-1) - p2.y) : (p1.y - (NY-1) - p2.y))
        : p1.y - p2.y;
    #else
    dy = p1.y - p2.y;
    #endif
    // Z direction
    #ifdef BC_Z_PERIODIC
    dz = abs(p1.z - p2.z) > ((NZ-1) / 2.0) ?
        (p1.z < p2.z ? (p1.z + (NZ-1) - p2.z) : (p1.z - (NZ-1) - p2.z))
        : p1.z - p2.z;
    #else
    dz = p1.z - p2.z;
    #endif
    return dfloat3(dx, dy, dz);
}


__device__
dfloat point_to_segment_distance_periodic(dfloat3 p, dfloat3 segA, dfloat3 segB, dfloat3 closestOnAB[1]) {
    dfloat minDist = 1E+37f;
    dfloat3 bestClosestOnAB;

    for (int i = 0; i < NUM_PERIODIC_DOMAIN_OFFSET; ++i) {
        int dx = PERIODIC_DOMAIN_OFFSET[i][0];
        int dy = PERIODIC_DOMAIN_OFFSET[i][1];
        int dz = PERIODIC_DOMAIN_OFFSET[i][2];

        // Translate the segment by the periodic offsets
        dfloat3 segA_translated = segA + dfloat3(dx * NX, dy * NY, dz * NZ);
        dfloat3 segB_translated = segB + dfloat3(dx * NX, dy * NY, dz * NZ);

        // Compute the closest point on the translated segment
        dfloat3 ab = segB_translated - segA_translated;
        dfloat3 ap = p - segA_translated;

        dfloat ab_dot_ab = dot_product(ab, ab);
        dfloat t = 0.0f;
        if (ab_dot_ab > 0) { //in case of zero-length segment
            t = dot_product(ap, ab) / ab_dot_ab;
        }

        t = myMax(0.0f, myMin(1.0f, t)); // Clamp t to [0, 1]

        dfloat3 tempClosestOnAB = segA_translated + ab * t;
        dfloat dist = vector_length(p - tempClosestOnAB);

        // Update the minimum distance and store the closest point
        if (dist < minDist) {
            minDist = dist;
            bestClosestOnAB = tempClosestOnAB;
        }
    }

    closestOnAB[0] = bestClosestOnAB;
    return minDist;
}

// Project a point onto a segment and constrain it within the segment
__device__
dfloat3 constrain_to_segment(dfloat3 point, dfloat3 segStart, dfloat3 segEnd) {
    dfloat3 segDir = segEnd - segStart;
    dfloat segLengthSqr = dot_product(segDir,segDir);
    if (segLengthSqr == 0.0) 
        return segStart;  // The segment is a point

    dfloat t = dot_product((point - segStart), segDir) / segLengthSqr;
    t = clamp01(t);

    return (segStart + (segDir * t));
}

// Main function to compute the closest distance between two segments and return the closest points
__device__
dfloat segment_segment_closest_points(dfloat3 p1, dfloat3 q1, dfloat3 p2, dfloat3 q2, dfloat3 closestOnAB[1], dfloat3 closestOnCD[1]) {

    dfloat3 segDC = (q2 - p2);  // Vector from p2 to q2 (segment [p2, q2])
    dfloat lineDirSqrMag = dot_product(segDC,segDC);  // Square magnitude of segment [p2, q2]

    // Project p1 and q1 onto the plane defined by segment [p2, q2]
    dfloat3 inPlaneA = p1 - ((dot_product(p1-p2,segDC)/lineDirSqrMag)*segDC);
    dfloat3 inPlaneB = q1 - ((dot_product(q1-p2,segDC)/lineDirSqrMag)*segDC);
    dfloat3 inPlaneBA = (inPlaneB - inPlaneA);
    dfloat t = dot_product(p2-inPlaneA,inPlaneBA) / dot_product(inPlaneBA, inPlaneBA);


    if (dot_product(inPlaneBA, inPlaneBA) == 0.0) {
        t = 0.0;  // Handle case where inPlaneA and inPlaneB are the same (segments are parallel)
    }

    // Find the closest point on segment [p1, q1] to the line [p2, q2]
    dfloat3 segABtoLineCD = p1 + clamp01(t)*(q1-p1);

    // Constrain the result to segment [p2, q2]
    closestOnCD[0] = constrain_to_segment(segABtoLineCD, p2, q2);

    // Constrain the closest point on segment [p2, q2] back to segment [p1, q1]
    closestOnAB[0] = constrain_to_segment(closestOnCD[0], p1, q1);


    // Calculate the distance between the closest points on the two segments
    dfloat3 diff = vector_length(closestOnAB[0] - closestOnCD[0]);
    return vector_length(diff);  // Return the distance between the closest points
}


__device__
dfloat segment_segment_closest_points_periodic(dfloat3 p1, dfloat3 q1, dfloat3 p2, dfloat3 q2, dfloat3 closestOnAB[1], dfloat3 closestOnCD[1]){
    dfloat minDist = 1E+37f;
    dfloat3 bestClosestOnAB, bestClosestOnCD;

    for (int i = 0; i < NUM_PERIODIC_DOMAIN_OFFSET; ++i) {
        int dx = PERIODIC_DOMAIN_OFFSET[i][0];
        int dy = PERIODIC_DOMAIN_OFFSET[i][1];
        int dz = PERIODIC_DOMAIN_OFFSET[i][2];

        dfloat3 p2_translated = p2 + dfloat3(dx * NX, dy * NY, dz * NZ);
        dfloat3 q2_translated = q2 + dfloat3(dx * NX, dy * NY, dz * NZ);

        dfloat3 tempClosestOnAB, tempClosestOnCD;
        dfloat dist = segment_segment_closest_points(p1, q1, p2_translated, q2_translated, &tempClosestOnAB, &tempClosestOnCD);

        if (dist < minDist) {
            minDist = dist;
            bestClosestOnAB = tempClosestOnAB;
            bestClosestOnCD = tempClosestOnCD;
        }
    }

    closestOnAB[0] = bestClosestOnAB;
    closestOnCD[0] = bestClosestOnCD;

    return minDist;
}


// ****************************************************************************
// ************************   MATRIX OPERATIONS   *****************************
// ****************************************************************************

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



// ****************************************************************************
// **********************   QUARTENION OPERATIONS   ***************************
// ****************************************************************************

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
dfloat4 quart_multiplication(dfloat4 q1, dfloat4 q2){
    dfloat4 q;
    
    q.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    q.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    q.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    q.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

    return q;
}


// ****************************************************************************
// **********************   CONVERSION OPERATIONS   ***************************
// ****************************************************************************

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
dfloat4 euler_to_quart(dfloat roll, dfloat pitch, dfloat yaw){
    dfloat cr = cos(roll * 0.5);
    dfloat sr = sin(roll * 0.5);
    dfloat cp = cos(pitch * 0.5);
    dfloat sp = sin(pitch * 0.5);
    dfloat cy = cos(yaw * 0.5);
    dfloat sy = sin(yaw * 0.5);

    dfloat4 q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

__host__ __device__
dfloat3 quart_to_euler(dfloat4 q){
    dfloat3 angles;

    // roll (x-axis rotation)
    dfloat sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
    dfloat cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
    angles.x = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    dfloat sinp = std::sqrt(1 + 2 * (q.w * q.y - q.x * q.z));
    dfloat cosp = std::sqrt(1 - 2 * (q.w * q.y - q.x * q.z));
    angles.y = 2 * std::atan2(sinp, cosp) - M_PI / 2;

    // yaw (z-axis rotation)
    dfloat siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    dfloat cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    angles.z = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}

__device__
void rotationMatrixFromVectors(dfloat3 v1, dfloat3 v2, dfloat R[3][3]){
    dfloat3 v3 = cross_product(v1,v2);

    R[0][0] = v1.x; R[1][0] = v2.x; R[2][0] = v3.x;
    R[0][1] = v1.y; R[1][1] = v2.y; R[2][1] = v3.y;
    R[0][2] = v1.z; R[1][2] = v2.z; R[2][2] = v3.z;

}

__device__
void rotationMatrixFromVectors(dfloat3 v1, dfloat3 v2, dfloat3 v3, dfloat R[3][3]){
    R[0][0] = v1.x; R[1][0] = v2.x; R[2][0] = v3.x;
    R[0][1] = v1.y; R[1][1] = v2.y; R[2][1] = v3.y;
    R[0][2] = v1.z; R[1][2] = v2.z; R[2][2] = v3.z;
}


// ****************************************************************************
// ***********************   ROTATION OPERATIONS   ****************************
// ****************************************************************************


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
    dfloat angle_d2 = acosf(dot)*0.5;

    // Calculate the axis of rotation
    dfloat3 axis = cross_product(v1, v2);
    axis = vector_normalize(axis);

    dfloat4 q;
    q.w = cosf(angle_d2 );
    q.x = axis.x * sinf(angle_d2);
    q.y = axis.y * sinf(angle_d2);
    q.z = axis.z * sinf(angle_d2);

    return q;
}

__host__ __device__
dfloat4 axis_angle_to_quart(dfloat3 axis, dfloat angle) {
    dfloat4 q;
    angle = angle*0.5;
    // Normalize the axis of rotation
    axis = vector_normalize(axis);
    
    // Compute the quaternion
    q.w = cosf(angle);
    q.x = axis.x * sinf(angle);
    q.y = axis.y * sinf(angle);
    q.z = axis.z * sinf(angle);
    
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

__host__ __device__
dfloat mom_trilinear_interp(dfloat x, dfloat y, dfloat z, const int mom , dfloat *fMom) {
    int i = (int)floor(x);
    int j = (int)floor(y);
    int k = (int)floor(z);


    dfloat xi = x - i;
    dfloat eta = y - j;
    dfloat zeta = z - k;

    // Direct value (no interpolation needed)
    if (xi == 0.0 && eta == 0.0 && zeta == 0.0) {
        return getMom(i, j, k, mom, fMom);
    }

    // Linear in x direction
    if (eta == 0.0 && zeta == 0.0) {
        dfloat c0 = getMom(i,     j, k, mom, fMom);
        dfloat c1 = getMom(i + 1, j, k, mom, fMom);
        return (1 - xi) * c0 + xi * c1;
    }

    // Linear in y direction
    if (xi == 0.0 && zeta == 0.0) {
        dfloat c0 = getMom(i, j,     k, mom, fMom);
        dfloat c1 = getMom(i, j + 1, k, mom, fMom);
        return (1 - eta) * c0 + eta * c1;
    }

    // Linear in z direction
    if (xi == 0.0 && eta == 0.0) {
        dfloat c0 = getMom(i, j, k,     mom, fMom);
        dfloat c1 = getMom(i, j, k + 1, mom, fMom);
        return (1 - zeta) * c0 + zeta * c1;
    }

    // Bilinear in xy plane (z fixed)
    if (zeta == 0.0) {
        dfloat c00 = getMom(i,     j,     k, mom, fMom);
        dfloat c10 = getMom(i + 1, j,     k, mom, fMom);
        dfloat c01 = getMom(i,     j + 1, k, mom, fMom);
        dfloat c11 = getMom(i + 1, j + 1, k, mom, fMom);
        return (1 - xi) * (1 - eta) * c00 +
                xi      * (1 - eta) * c10 +
               (1 - xi) * eta       * c01 +
                xi      * eta       * c11;
    }

    // Bilinear in xz plane (y fixed)
    if (eta == 0.0) {
        dfloat c00 = getMom(i,     j, k,     mom, fMom);
        dfloat c10 = getMom(i + 1, j, k,     mom, fMom);
        dfloat c01 = getMom(i,     j, k + 1, mom, fMom);
        dfloat c11 = getMom(i + 1, j, k + 1, mom, fMom);
        return (1 - xi) * (1 - zeta) * c00 +
                xi      * (1 - zeta) * c10 +
               (1 - xi) * zeta       * c01 +
                xi      * zeta       * c11;
    }

    // Bilinear in yz plane (x fixed)
    if (xi == 0.0) {
        dfloat c00 = getMom(i, j,     k,     mom, fMom);
        dfloat c10 = getMom(i, j + 1, k,     mom, fMom);
        dfloat c01 = getMom(i, j,     k + 1, mom, fMom);
        dfloat c11 = getMom(i, j + 1, k + 1, mom, fMom);
        return (1 - eta) * (1 - zeta) * c00 +
                eta      * (1 - zeta) * c10 +
               (1 - eta) * zeta       * c01 +
                eta      * zeta       * c11;
    }

    // Full trilinear
    dfloat c000 = getMom(i,     j,     k,     mom, fMom);
    dfloat c100 = getMom(i + 1, j,     k,     mom, fMom);
    dfloat c010 = getMom(i,     j + 1, k,     mom, fMom);
    dfloat c110 = getMom(i + 1, j + 1, k,     mom, fMom);
    dfloat c001 = getMom(i,     j,     k + 1, mom, fMom);
    dfloat c101 = getMom(i + 1, j,     k + 1, mom, fMom);
    dfloat c011 = getMom(i,     j + 1, k + 1, mom, fMom);
    dfloat c111 = getMom(i + 1, j + 1, k + 1, mom, fMom);

    return
        (1 - xi) * (1 - eta) * (1 - zeta) * c000 +
         xi      * (1 - eta) * (1 - zeta) * c100 +
        (1 - xi) * eta       * (1 - zeta) * c010 +
         xi      * eta       * (1 - zeta) * c110 +
        (1 - xi) * (1 - eta) * zeta       * c001 +
         xi      * (1 - eta) * zeta       * c101 +
        (1 - xi) * eta       * zeta       * c011 +
         xi      * eta       * zeta       * c111;
}



__host__ __device__
dfloat cubic_interp(dfloat p0, dfloat p1, dfloat p2, dfloat p3, dfloat t) {
    dfloat a = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    dfloat b = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    dfloat c = -0.5*p0 + 0.5*p2;
    dfloat d = p1;
    return ((a*t + b)*t + c)*t + d;
}

__host__ __device__
dfloat mom_tricubic_interp(dfloat x, dfloat y, dfloat z,
                       const int mom, dfloat *fMom) {
    int i = (int)floor(x);
    int j = (int)floor(y);
    int k = (int)floor(z);

    dfloat xi   = x - i;
    dfloat eta  = y - j;
    dfloat zeta = z - k;

    dfloat F[4][4][4];

    // collect 64 neighbors
    for (int kk = 0; kk < 4; kk++) {
        for (int jj = 0; jj < 4; jj++) {
            for (int ii = 0; ii < 4; ii++) {
                F[ii][jj][kk] = getMom(i + ii - 1, j + jj - 1, k + kk - 1, mom, fMom);
            }
        }
    }

    dfloat C[4][4];  // after x interpolation
    dfloat D[4];     // after y interpolation

    // interpolate in x for each row
    for (int jj = 0; jj < 4; jj++) {
        for (int kk = 0; kk < 4; kk++) {
            C[jj][kk] = cubic_interp(F[0][jj][kk], F[1][jj][kk], F[2][jj][kk], F[3][jj][kk], xi);
        }
    }

    // interpolate in y for each z
    for (int kk = 0; kk < 4; kk++) {
        D[kk] = cubic_interp(C[0][kk], C[1][kk], C[2][kk], C[3][kk], eta);
    }

    // interpolate in z
    return cubic_interp(D[0], D[1], D[2], D[3], zeta);
}
