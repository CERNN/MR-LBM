//functions that determine HOW colide
#include "collision.cuh"

#ifdef PARTICLE_MODEL

//collision tracking
__device__ 
int calculateWallIndex(const dfloat3 &n) {
    // Calculate the index based on the normal vector
    return 7 + (1 - (int)n.x) - 2 * (1 - (int)n.y) - 3 * (1 - (int)n.z);
    /*
    n.x n.y n.z index
    1   0   0   2
    -1  0   0   4
    0   1   0   5
    0   -1  0   1
    0   0   1   6
    0   0   -1  0
    0   0   0   3 -> external duct, since normal will be reduced to 0,0,0 when convert to integer
    */
}

__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep) {
    for (int i = 7; i < MAX_ACTIVE_COLLISIONS ; i++) {
        if (collisionData.getCollisionPartnerID(i) == partnerID &&
            currentTimeStep - collisionData.getLastCollisionStep(i) <= 1) {
            return i; // Found the collision index for a particle
        }
    }

    // If no match is found, return -1
    return -1;
}
__device__ 
int startCollision(CollisionData &collisionData, int partnerID, bool isWall, const dfloat3 &wallNormal, int currentTimeStep) {
    int index = -1;
    if (isWall) {
        index = calculateWallIndex(wallNormal);
        // Initialize wall collision data
        collisionData.setTangentialDisplacement(index, {0.0, 0.0, 0.0});
        collisionData.setLastCollisionStep(index, currentTimeStep);
    } else {
        for (int i = 7; i < MAX_ACTIVE_COLLISIONS; i++) {
            if (collisionData.getLastCollisionStep(i) == -1) { // Check for an unused slot
                collisionData.setCollisionPartnerID(i, partnerID);
                collisionData.setTangentialDisplacement(i, {0.0, 0.0, 0.0});
                collisionData.setLastCollisionStep(i, currentTimeStep);
                index = i;
                break;
            }
        }
    }

    return index;
}
__device__ 
dfloat3 updateTangentialDisplacement(CollisionData &collisionData, int index, const dfloat3 &displacement, int currentTimeStep) {
        collisionData.setTangentialDisplacement(index, (collisionData.getTangentialDisplacement(index) + displacement));
        collisionData.setLastCollisionStep(index, currentTimeStep);

    return collisionData.getTangentialDisplacement(index);
}
__device__ 
void    endCollision(CollisionData &collisionData, int index, int currentTimeStep) {
    // Check if index is valid for wall collisions (0 to 6)
    if (index >= 0 && index <= 6) {
        collisionData.setLastCollisionStep(index, -1);
    }
    // Check if index is valid for particle collisions 
    else if (index >= 7 && index < MAX_ACTIVE_COLLISIONS ) {
        if (currentTimeStep - collisionData.getLastCollisionStep(index) > 1) {
            collisionData.setTangentialDisplacement(index, {0.0, 0.0, 0.0});
            collisionData.setLastCollisionStep(index, -1); // Indicate the slot is available 
            collisionData.setCollisionPartnerID(index, -1); // Optionally reset
        }
    }
}

//collision mechanics with walls

__device__
void sphereWallCollision(ParticleCenter* pc_i,Wall wallData,dfloat displacement,int step){

    //particle information
    const dfloat m_i = pc_i->getVolume() * pc_i ->getDensity();
    const dfloat r_i = pc_i->getRadius();

    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();

    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector
    dfloat3 n = wallData.normal;

    //invert collision direction since is from sphere to wall
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;

    //relative velocity
    dfloat3 G = v_i - wall_speed;

    //constants 
    //effective radius and mass
    const dfloat effective_radius = r_i; //wall is r = infinity
    const dfloat effective_mass = m_i; //wall has infinite mass
    //collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(effective_radius));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PW_REST_COEF)  / (sqrt(M_PI*M_PI + log(PW_REST_COEF)*log(PW_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal;
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25); 
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25); 
    dfloat f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force
    dfloat3 G_ct; //relative tangential velocity
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;
    
    dfloat mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    dfloat3 t;//tangential velocity vector
    if (mag != 0){
        //tangential vector
        t.x = G_ct.x/mag;
        t.y = G_ct.y/mag;
        t.z = G_ct.z/mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }

    //retrive stored displacedment 
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp; //total tangential displacement

    int last_step = pc_i->getCollision().getLastCollisionStep(tang_index);
    if(step - last_step > 1){ //there is no prior collision
        //first need to erase previous collision
        endCollision(pc_i->getCollision(),tang_index,step);
        //now we start the new collision tracking
        startCollision(pc_i->getCollision(),tang_index,true,n,step);
        tang_disp = G_ct;
    }else{//there is already a collision in progress
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,G_ct,step);
    }
    

    //tangential force
    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    //determine if slip or not
    if(  mag > PW_FRICTION_COEF * fabsf(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,-G_ct,step);
        f_tang.x = - PW_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PW_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PW_FRICTION_COEF * f_n * t.z;
    }

    //sum the forces
    dfloat3 f_dirs = dfloat3(
        f_normal.x + f_tang.x,
        f_normal.y + f_tang.y,
        f_normal.z + f_tang.z
    );

    //calculate moments
    dfloat3 m_dirs = dfloat3(
        r_i * (n.y*f_tang.z - n.z*f_tang.y),
        r_i * (n.z*f_tang.x - n.x*f_tang.z),
        r_i * (n.x*f_tang.y - n.y*f_tang.x)
    );

    //save data in the particle information
    atomicAdd(&(pc_i->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), f_dirs.z);

    atomicAdd(&(pc_i->getMXatomic()), m_dirs.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs.z);
}
__device__
void capsuleWallCollisionCap(ParticleCenter* pc_i,Wall wallData,dfloat displacement,dfloat3 endpoint, int step){

    //particle information
    const dfloat3 pos_i = pc_i->getPos(); //center position
    const dfloat3 pos_c_i = endpoint; //cap position

    const dfloat m_i = pc_i ->getVolume() * pc_i ->getDensity();
    const dfloat r_i = pc_i->getRadius();

    const dfloat3 v_i = pc_i->getVel(); //VELOCITY OF THE CENTER OF MASS
    const dfloat3 w_i = pc_i->getW();

    //wall information        
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector
    dfloat3 n = wallData.normal;

    //invert collision direction since is from sphere to wall
    n.x = -n.x;
    n.y = -n.y;
    n.z = -n.z;


    //vector center-> cap
    dfloat3 rr = pos_c_i - pos_i;

    dfloat3 G = v_i - wall_speed;


    //constants 
    //effective radius and mass
    const dfloat effective_radius = r_i; //wall is r = infinity
    const dfloat effective_mass = m_i; //wall has infinite mass
    //collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(effective_radius));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PW_REST_COEF)  / (sqrt(M_PI*M_PI + log(PW_REST_COEF)*log(PW_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal = f_kn * n - DAMPING_NORMAL *  dot_product(G,n) * n * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n = vector_length(f_normal);

    //tangential force
    dfloat3 G_ct = G + r_i * cross_product(w_i,n+rr) - dot_product(G,n)*n;
    dfloat mag = vector_length(G_ct);


    dfloat3 t;//tangential velocity vector
    if (mag != 0){
        //tangential vector
        t = G_ct / mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }

    //retrive stored displacedment 
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp; //total tangential displacement

    int last_step = pc_i->getCollision().getLastCollisionStep(tang_index);
    if(step - last_step > 1){ //there is no prior collision
        //first need to erase previous collision
        endCollision(pc_i->getCollision(),tang_index,step);
        //now we start the new collision tracking
        startCollision(pc_i->getCollision(),tang_index,true,n,step);
        tang_disp = G_ct;
    }else{//there is already a collision in progress
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,G_ct,step);
    }

    //tangential force
    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = vector_length(f_tang);

    //determine if slip or not,
    if(  mag > PW_FRICTION_COEF * fabsf(f_n) ){
         tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,-G_ct,step);
        f_tang = - PW_FRICTION_COEF * f_n * t;
    }

    //sum the forces
    dfloat3 f_dirs = f_normal + f_tang;

    //calculate moments
    dfloat3 m_dirs = cross_product((n*r_i) + rr,f_dirs);

    //save date in the particle information
    atomicAdd(&(pc_i->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), f_dirs.z);

    atomicAdd(&(pc_i->getMXatomic()), m_dirs.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs.z);
    

}

__device__
dfloat sphereSphereGap(ParticleCenter*  pc_i, ParticleCenter*  pc_j) {
    dfloat3 p1 = pc_i->getPos();
    dfloat3 p2 = pc_j->getPos();

    dfloat r1 = pc_i->getRadius();
    dfloat r2 = pc_j->getRadius();

    dfloat3 delta = p1 - p2;

    #ifdef BC_X_PERIODIC
        if(delta.x > NX / 2.0) delta.x -= NX;
        if(delta.x < -NX / 2.0) delta.x += NX;
    #endif
    #ifdef BC_Y_PERIODIC
        if(delta.y > NY / 2.0) delta.y -= NY;
        if(delta.y < -NY / 2.0) delta.y += NY;
    #endif
    #ifdef BC_Z_PERIODIC
        if(delta.z > NZ / 2.0) delta.z -= NZ;
        if(delta.z < -NZ / 2.0) delta.z += NZ;
    #endif

    dfloat dist = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    
    return dist - (r1 + r2);
}

__device__
dfloat point_to_segment_distance_periodic(dfloat3 p, dfloat3 segA, dfloat3 segB, dfloat3 closestOnAB[1]) {
    dfloat minDist = 1E+37;  // Initialize to a large value
    dfloat3 bestClosestOnAB;
    int dx = 0, dy = 0, dz = 0;

    // Loop over periodic offsets in x, y, and z if periodic boundary conditions are enabled
    #ifdef BC_X_PERIODIC
    for (dx = -1; dx <= 1; dx++) {
    #endif
        #ifdef BC_Y_PERIODIC
        for (dy = -1; dy <= 1; dy++) {
        #endif
            #ifdef BC_Z_PERIODIC
            for (dz = -1; dz <= 1; dz++) {
            #endif
                // Translate the segment by the periodic offsets
                dfloat3 segA_translated = segA + dfloat3(dx * NX, dy * NY, dz * NZ);
                dfloat3 segB_translated = segB + dfloat3(dx * NX, dy * NY, dz * NZ);

                // Compute the closest point on the translated segment
                dfloat3 ab = segB_translated - segA_translated;
                dfloat3 ap = p - segA_translated;
                dfloat t = dot_product(ap, ab) / dot_product(ab, ab);
                t = myMax(0, myMin(1, t));  // Clamp t to [0, 1]

                dfloat3 tempClosestOnAB = segA_translated + ab * t;
                dfloat dist = vector_length(p - tempClosestOnAB);

                // Update the minimum distance and store the closest point
                if (dist < minDist) {
                    minDist = dist;
                    bestClosestOnAB = tempClosestOnAB;
                }

            #ifdef BC_Z_PERIODIC
            } // End Z loop
            #endif
        #ifdef BC_Y_PERIODIC
        } // End Y loop
        #endif
    #ifdef BC_X_PERIODIC
    } // End X loop
    #endif

    // Store the closest point on the segment
    closestOnAB[0] = bestClosestOnAB;

    // Return the minimum distance
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
    dfloat minDist = 1E+37;  // Initialize to a large value
    dfloat3 bestClosestOnAB, bestClosestOnCD;
    int dx = 0;
    int dy = 0;
    int dz = 0;
    #ifdef BC_X_PERIODIC
    for ( dx = -1; dx <= 1; dx++) {
    #endif
        #ifdef BC_Y_PERIODIC
        for ( dy = -1; dy <= 1; dy++) {
        #endif
            #ifdef BC_Z_PERIODIC
            for ( dz = -1; dz <= 1; dz++) {
            #endif
                // Translate segment [p2, q2] by periodic offsets
                dfloat3 p2_translated = p2 + dfloat3(dx * (NX-1), dy * (NY-1), dz * (NZ-1));
                dfloat3 q2_translated = q2 + dfloat3(dx * (NX-1), dy * (NY-1), dz * (NZ-1));

                // Compute closest points between segment [p1, q1] and translated segment [p2_translated, q2_translated]
                dfloat3 tempClosestOnAB, tempClosestOnCD;
                dfloat dist = segment_segment_closest_points(p1, q1, p2_translated, q2_translated, &tempClosestOnAB, &tempClosestOnCD);
                // Update minimum distance and store the best closest points
                if (dist < minDist) {
                    minDist = dist;
                    bestClosestOnAB = tempClosestOnAB;
                    bestClosestOnCD = tempClosestOnCD;
                }

            #ifdef BC_Z_PERIODIC
            }
            #endif
        #ifdef BC_Y_PERIODIC
        }
        #endif

    #ifdef BC_X_PERIODIC
    }
    #endif
    closestOnAB[0] = bestClosestOnAB;
    closestOnCD[0] = bestClosestOnCD;

    return minDist;  // Return the minimum distance between the segments
}

// ------------------------------------------------------------------------ 
// -------------------- SPHERE COLLISION ---------- -----------------------
// ------------------------------------------------------------------------ 

__device__
void sphereSphereCollision(unsigned int column,unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step){
    // Particle i info (column)
    const dfloat3 pos_i = pc_i->getPos();
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i ->getVolume() * pc_i ->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
   
    // Particle j info (row)
    const dfloat3 pos_j = pc_j->getPos();
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j ->getVolume() * pc_j ->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();



    //first check if they will collide
    const dfloat3 diff_pos = dfloat3(
        #ifdef BC_X_WALL
            pos_i.x - pos_j.x
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC 
        abs(pos_i.x - pos_j.x) > ((BC_X_E - BC_X_0) / 2.0) ? 
        (pos_i.x < pos_j.x ?
            (pos_i.x + (BC_X_E - BC_X_0) - pos_j.x)
            : 
            (pos_i.x - (BC_X_E - BC_X_0) - pos_j.x)
        )
        : pos_i.x - pos_j.x
        #endif //BC_X_PERIODIC
        , 
        #ifdef BC_Y_WALL
           (pos_i.y - pos_j.y)
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
        abs(pos_i.y - pos_j.y) > ((BC_Y_E - BC_Y_0) / 2.0) ? 
        (pos_i.y < pos_j.y ?
            (pos_i.y + (BC_Y_E - BC_Y_0) - pos_j.y)
            : 
            (pos_i.y - (BC_Y_E - BC_Y_0) - pos_j.y)
        )
        : pos_i.y - pos_j.y
        #endif //BC_Y_PERIODIC
        , 
        #ifdef BC_Z_WALL
            pos_i.z - pos_j.z
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            abs(pos_i.z - pos_j.z) > ((BC_Z_E - BC_Z_0) / 2.0) ? 
            (pos_i.z < pos_j.z ?
                (pos_i.z + (BC_Z_E - BC_Z_0) - pos_j.z)
                : 
                (pos_i.z - (BC_Z_E - BC_Z_0) - pos_j.z)
            )
            : pos_i.z - pos_j.z
        #endif //BC_Z_PERIODIC
    );

    const dfloat mag_dist = sqrt(
        diff_pos.x*diff_pos.x
        + diff_pos.y*diff_pos.y
        + diff_pos.z*diff_pos.z);

    if(mag_dist > r_i+r_j) //they dont collide
        return;

    //but if they collide, we can do some calculations

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    //normal deformation
    dfloat displacement = r_i + r_j - mag_dist;
    // relative velocity vector
    dfloat3 G = v_i-v_j;

    //HERTZ CONTACT THEORY

    dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal;
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25);
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n;
    f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force
    dfloat3 G_ct;       
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) + r_j*(w_j.y*n.z - w_j.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) + r_j*(w_j.z*n.x - w_j.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) + r_j*(w_j.x*n.y - w_j.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;

    dfloat mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    //calculate tangential vector
    dfloat3 t;
    if (mag != 0){
        //tangential vector
        t.x = G_ct.x/mag;
        t.y = G_ct.y/mag;
        t.z = G_ct.z/mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }

    //retrive stored displacedment 
    dfloat3 tang_disp; //total tangential displacement
    int tang_index = getCollisionIndexByPartnerID(pc_i->getCollision(),row,step);
    if(tang_index == -1){ //no previous collision was detected
        tang_index = startCollision(pc_i->getCollision(),row,false,dfloat3(0,0,0),step);
        tang_disp = G_ct;
    }else{
        //check if the collision already exited in the past
        if(step - pc_i->getCollision().getLastCollisionStep(tang_index) > 1){ //already existed but ended
            endCollision(pc_i->getCollision(),tang_index,step); //end current one
            tang_index = startCollision(pc_i->getCollision(),row,false,dfloat3(0,0,0),step);
            tang_disp = G_ct;
        }else{ //collision is still ongoing
            tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,G,step);
        }
    }

    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,-G_ct,step);
        f_tang.x = - PP_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PP_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PP_FRICTION_COEF * f_n * t.z;
    }
    //FINAL FORCE RESULTS


    // Force in each direction
    dfloat3 f_dirs = dfloat3(
        f_normal.x + f_tang.x,
        f_normal.y + f_tang.y,
        f_normal.z + f_tang.z
    );
    //Torque in each direction
    dfloat3 m_dirs_i = dfloat3(
        r_i * (n.y*f_tang.z - n.z*f_tang.y),
        r_i * (n.z*f_tang.x - n.x*f_tang.z),
        r_i * (n.x*f_tang.y - n.y*f_tang.x)
    );
    dfloat3 m_dirs_j = dfloat3(
        r_j * (n.y*f_tang.z - n.z*f_tang.y),
        r_j * (n.z*f_tang.x - n.x*f_tang.z),
        r_j * (n.x*f_tang.y - n.y*f_tang.x)
    );

    // Force positive in particle i (column)
    atomicAdd(&(pc_i->getFXatomic()), -f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), -f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), -f_dirs.z);

    atomicAdd(&(pc_i->getMXatomic()), m_dirs_i.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs_i.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs_i.z);

    // Force negative in particle j (row)
    atomicAdd(&(pc_j->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_j->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_j->getFZatomic()), f_dirs.z);

    atomicAdd(&(pc_j->getMXatomic()), m_dirs_j.x); //normal vector takes care of negative sign
    atomicAdd(&(pc_j->getMYatomic()), m_dirs_j.y);
    atomicAdd(&(pc_j->getMZatomic()), m_dirs_j.z); 


}

// ------------------------------------------------------------------------ 
// -------------------- CAPSULE COLLISIONS -------- -----------------------
// ------------------------------------------------------------------------ 

__device__
void capsuleCapsuleCollision(unsigned int column, unsigned int row, ParticleCenter* pc_i,  ParticleCenter* pc_j, dfloat3 closestOnA[1], dfloat3 closestOnB[1], int step){
    // Particle i info (column)
    const dfloat3 pos_i = closestOnA[0];
    const dfloat3 pos_c_i = pc_i->getPos();
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i ->getVolume() * pc_i ->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
   
    // Particle j info (row)
    const dfloat3 pos_j =closestOnB[0];
    const dfloat3 pos_c_j = pc_j->getPos();
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j ->getVolume() * pc_j ->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();



    //first check if they will collide
    const dfloat3 diff_pos = dfloat3(
        #ifdef BC_X_WALL
            pos_i.x - pos_j.x
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC 
        abs(pos_i.x - pos_j.x) > ((BC_X_E - BC_X_0) / 2.0) ? 
        (pos_i.x < pos_j.x ?
            (pos_i.x + (BC_X_E - BC_X_0) - pos_j.x)
            : 
            (pos_i.x - (BC_X_E - BC_X_0) - pos_j.x)
        )
        : pos_i.x - pos_j.x
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL
            pos_i.y - pos_j.y
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
        abs(pos_i.y - pos_j.y) > ((BC_Y_E - BC_Y_0) / 2.0) ? 
        (pos_i.y < pos_j.y ?
            (pos_i.y + (BC_Y_E - BC_Y_0) - pos_j.y)
            : 
            (pos_i.y - (BC_Y_E - BC_Y_0) - pos_j.y)
        )
        : pos_i.y - pos_j.y
        #endif //BC_Y_PERIODIC
        ,
        #ifdef BC_Z_WALL
            pos_i.z - pos_j.z
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            abs(pos_i.z - pos_j.z) > ((BC_Z_E - BC_Z_0) / 2.0) ? 
            (pos_i.z < pos_j.z ?
                (pos_i.z + (BC_Z_E - BC_Z_0) - pos_j.z)
                : 
                (pos_i.z - (BC_Z_E - BC_Z_0) - pos_j.z)
            )
            : pos_i.z - pos_j.z
        #endif //BC_Z_PERIODIC
    );

    const dfloat mag_dist = sqrt(
          diff_pos.x*diff_pos.x
        + diff_pos.y*diff_pos.y
        + diff_pos.z*diff_pos.z);

    if(mag_dist > r_i+r_j) //they dont collide
        return;

    //but if they collide, we can do some calculations

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    //normal deformation
    dfloat displacement = r_i + r_j - mag_dist;
    // relative velocity vector
    dfloat3 G = v_i-v_j;

    //HERTZ CONTACT THEORY

    dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);

    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal;
    f_normal.x = f_kn * n.x - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.x * POW_FUNCTION(abs(displacement),0.25);
    f_normal.y = f_kn * n.y - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.y * POW_FUNCTION(abs(displacement),0.25);
    f_normal.z = f_kn * n.z - DAMPING_NORMAL * (G.x*n.x + G.y*n.y + G.z*n.z)*n.z * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n;
    f_n = sqrt(f_normal.x*f_normal.x + f_normal.y*f_normal.y + f_normal.z*f_normal.z);

    //tangential force
    dfloat3 G_ct;       
    G_ct.x = G.x + r_i*(w_i.y*n.z - w_i.z*n.y) + r_j*(w_j.y*n.z - w_j.z*n.y) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.x;
    G_ct.y = G.y + r_i*(w_i.z*n.x - w_i.x*n.z) + r_j*(w_j.z*n.x - w_j.x*n.z) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.y;
    G_ct.z = G.z + r_i*(w_i.x*n.y - w_i.y*n.x) + r_j*(w_j.x*n.y - w_j.y*n.x) - (G.x*n.x + G.y*n.y + G.z*n.z) * n.z;

    dfloat mag = G_ct.x*G_ct.x+G_ct.y*G_ct.y+G_ct.z*G_ct.z;
    mag=sqrt(mag);

    //calculate tangential vector
    dfloat3 t;
    if (mag != 0){
        //tangential vector
        t.x = G_ct.x/mag;
        t.y = G_ct.y/mag;
        t.z = G_ct.z/mag;
    }else{
        t.x = 0.0;
        t.y = 0.0;
        t.z = 0.0;
    }

    //retrive stored displacedment 
    dfloat3 tang_disp; //total tangential displacement
    int tang_index = getCollisionIndexByPartnerID(pc_i->getCollision(),row,step);
    if(tang_index == -1){ //no previous collision was detected
        tang_index = startCollision(pc_i->getCollision(),row,false,dfloat3(0,0,0),step);
        tang_disp = G_ct;
    }else{
        //check if the collision already exited in the past
        if(step - pc_i->getCollision().getLastCollisionStep(tang_index) > 1){ //already existed but ended
            endCollision(pc_i->getCollision(),tang_index,step); //end current one
            tang_index = startCollision(pc_i->getCollision(),row,false,dfloat3(0,0,0),step);
            tang_disp = G_ct;
        }else{ //collision is still ongoing
            tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,G,step);
        }
    }

    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = sqrt(f_tang.x*f_tang.x + f_tang.y*f_tang.y + f_tang.z*f_tang.z);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,-G_ct,step);
        f_tang.x = - PP_FRICTION_COEF * f_n * t.x;
        f_tang.y = - PP_FRICTION_COEF * f_n * t.y;
        f_tang.z = - PP_FRICTION_COEF * f_n * t.z;
    }

    //FINAL FORCE RESULTS

    //printf("pp  step %d fny %f fnt %f fnz %f \n",step,f_normal.x,f_tang.y, f_normal.z);

    // Force in each direction
    dfloat3 f_dirs = f_normal + f_tang;
    //Torque in each direction
    dfloat3 m_dirs_i = cross_product((pos_i-pos_c_i) + (-n*r_i) ,-f_dirs);
    dfloat3 m_dirs_j = cross_product((pos_j-pos_c_j) + ( n*r_j) , f_dirs);
    
    // Force positive in particle i (column)
    atomicAdd(&(pc_i->getFXatomic()), -f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), -f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), -f_dirs.z);

    atomicAdd(&(pc_i->getMXatomic()), m_dirs_i.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs_i.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs_i.z);

    // Force negative in particle j (row)
    atomicAdd(&(pc_j->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_j->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_j->getFZatomic()), f_dirs.z);

    atomicAdd(&(pc_j->getMXatomic()), m_dirs_j.x); //normal vector takes care of negative sign
    atomicAdd(&(pc_j->getMYatomic()), m_dirs_j.y);
    atomicAdd(&(pc_j->getMZatomic()), m_dirs_j.z); 
}
#endif //PARTICLE_MODEL