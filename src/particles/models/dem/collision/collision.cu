//functions that determine HOW colide
#include "collision.cuh"

#ifdef PARTICLE_MODEL

// ****************************************************************************
// ************************   FORCE COMPUTATION   *****************************
// ****************************************************************************

__device__ 
dfloat3 computeNormalForce(const dfloat3& n, const dfloat3& G, dfloat displacement, dfloat stiffness, dfloat damping) {
    dfloat f_kn = -stiffness * sqrt(abs(displacement*displacement*displacement));
    return f_kn * n - damping * dot_product(G, n) * n * POW_FUNCTION(abs(displacement), 0.25);
}

__device__ dfloat3 computeTangentialForce(
    dfloat3& tang_disp, // will be updated if slip occurs
    const dfloat3& G_ct,
    dfloat stiffness,
    dfloat damping,
    dfloat friction_coef,
    dfloat f_n,
    const dfloat3& t,
    ParticleCenter* pc_i,
    int tang_index,
    int step
) {
    dfloat3 f_tang = -stiffness * tang_disp - damping * G_ct * POW_FUNCTION(abs(vector_length(tang_disp)), 0.25);
    f_tang.x = - stiffness * tang_disp.x - damping * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - stiffness * tang_disp.y - damping * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - stiffness * tang_disp.z - damping * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);
    dfloat mag = vector_length(f_tang);
    if (mag > friction_coef * fabsf(f_n)) {
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(), tang_index, -G_ct, step);
        f_tang = -friction_coef * f_n * t;
    }
    return f_tang;
}

__device__ void accumulateForceAndTorque(ParticleCenter* pc_i, const dfloat3& f_dirs, const dfloat3& m_dirs) {
    atomicAdd(&(pc_i->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), f_dirs.z);
    atomicAdd(&(pc_i->getMXatomic()), m_dirs.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs.z);
}



// ****************************************************************************
// ************************   COLLISION TRACKING   ****************************
// ****************************************************************************

__device__ 
int calculateWallIndex(const dfloat3 &n) {
    // Calculate the index based on the normal vector

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

    return 7 + (1 - (int)n.x) - 2 * (1 - (int)n.y) - 3 * (1 - (int)n.z);
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
void endCollision(CollisionData &collisionData, int index, int currentTimeStep) {
    // Check if index is valid for wall collisions (0 to 6)
    if (index >= 0 && index <= 6) {
        collisionData.setLastCollisionStep(index, -1);
    }
    // Check if index is valid for particle collisions 
    else if (index >= 7 && index < MAX_ACTIVE_COLLISIONS ) {
        if (currentTimeStep - collisionData.getLastCollisionStep(index) > 1) {
            collisionData.setTangentialDisplacement(index, {0.0, 0.0, 0.0});
            collisionData.setLastCollisionStep(index, -1); // Indicate the slot is available '
            collisionData.setCollisionPartnerID(index, -1); // Optionally reset
        }
    }
}


__device__ 
dfloat3 getOrUpdateTangentialDisplacement(
    ParticleCenter* pc_i,
    int identifier, // wall index or partner ID
    bool isWall, //true if wall, false if particle-particle collision
    int step,
    const dfloat3& G_ct,
    const dfloat3& G,
    int& tang_index_out,
    const dfloat3& wallNormal // only used for wall
) {
    dfloat3 tang_disp;
    int tang_index = -1;
    
    //get the collision index 
    if (isWall) {
        tang_index = calculateWallIndex(wallNormal);
    } else {
        tang_index = getCollisionIndexByPartnerID(pc_i->getCollision(), identifier, step);
    }

    //check if there is a current collision
    if (tang_index == -1) {
        tang_index = startCollision(pc_i->getCollision(), identifier, isWall, wallNormal, step);
        tang_disp = G_ct;
    } else {
        //check if the collision already exited in the past
        if (step - pc_i->getCollision().getLastCollisionStep(tang_index) > 1) {
            endCollision(pc_i->getCollision(), tang_index, step); //erase previous collision
            tang_index = startCollision(pc_i->getCollision(), identifier, isWall, wallNormal, step); //start collision and retrive collision index
            tang_disp = G_ct;
        } else {
            tang_disp = updateTangentialDisplacement(pc_i->getCollision(), tang_index, G, step);
        }
    }
    //return the index by address
    tang_index_out = tang_index;
    //return the tangential displacement
    return tang_disp;
}


// ****************************************************************************
// **************************   WALL COLLISION   ******************************
// ****************************************************************************

// -------------------------- SPHERE COLLISION --------------------------------
__device__
void sphereWallCollision(const CollisionContext& ctx){

    ParticleCenter* pc_i = ctx.pc_i;
    Wall wallData = ctx.wall;
    dfloat displacement = ctx.displacement;
    int step = ctx.step;

    // Particle info
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat r_i = pc_i->getRadius();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();

    // Wall info
    dfloat3 wall_speed = dfloat3(0,0,0);
    dfloat3 n = wallData.normal;
    n = n * -1.0f; //invert collision direction since is from sphere to wall

    // Relative velocity
    dfloat3 G = v_i - wall_speed;

    //Collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(r_i));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(r_i) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n) - dot_product(G, n) * n;
    dfloat mag = vector_length(G_ct);
    dfloat3 t = (mag != 0) ? (G_ct / mag) : dfloat3{0.0, 0.0, 0.0}; //tangential velocity vector


    //retrive and update tangential displacement
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, 0, true, step, G_ct, G, tang_index, n);

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, t, pc_i, tang_index, step
    );

    //Total forces and torque in the particle
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs = r_i * cross_product(n, f_tang);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs);
}

// ------------------------- CAPSULE COLLISIONS -------------------------------
__device__
void capsuleWallCollisionCap(const CollisionContext& ctx) {
    ParticleCenter* pc_i = ctx.pc_i;
    Wall wallData = ctx.wall;
    dfloat displacement = ctx.displacement;
    int step = ctx.step;
    dfloat3 endpoint = ctx.contactPoint;
    // Particle info
    const dfloat3 center_pos = pc_i->getPos();      // Particle center position
    const dfloat3 cap_pos = endpoint;               // Capsule cap position
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat r_i = pc_i->getRadius();
    const dfloat3 v_i = pc_i->getVel();             // Center of mass velocity
    const dfloat3 w_i = pc_i->getW();
    // Wall info
    dfloat3 wall_speed = dfloat3(0, 0, 0);          // Wall velocity (assumed zero)
    dfloat3 n = wallData.normal * -1.0f;            // Invert normal: collision from sphere to wall

    
    // Relative velocity
    dfloat3 rr = cap_pos - center_pos;              // Vector from center to cap
    dfloat3 G = v_i - wall_speed;                   // Relative velocity

    //Collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(r_i));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(r_i) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n + rr) - dot_product(G, n) * n;
    dfloat mag = vector_length(G_ct);
    dfloat3 t = (mag != 0) ? (G_ct / mag) : dfloat3{0.0, 0.0, 0.0}; // Tangential velocity vector

    // Tangential displacement tracking
    //retrive and update tangential displacement
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, 0, true, step, G_ct, G, tang_index, n);


    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, t, pc_i, tang_index, step
    );

    //Total forces and torque in the particle
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs = cross_product((n * r_i) + rr, f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs);
}

// ------------------------ ELLIPSOID COLLISIONS ------------------------------
__device__
void ellipsoidWallCollision(const CollisionContext& ctx, dfloat cr[1]) {
    ParticleCenter* pc_i = ctx.pc_i;
    Wall wallData = ctx.wall;
    dfloat displacement = ctx.displacement;
    int step = ctx.step;
    dfloat3 endpoint = ctx.contactPoint;
    // Particle info
    const dfloat3 pos_i = pc_i->getPos(); //center position
    const dfloat3 pos_c_i = endpoint; //contact point + n * radius of contact
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat r_i = pc_i->getRadius(); //TODO: find a way to calculate the correct radius of contact
    const dfloat3 v_i = pc_i->getVel(); //VELOCITY OF THE CENTER OF MASS
    const dfloat3 w_i = pc_i->getW();
    // Wall info
    dfloat3 wall_speed = dfloat3(0,0,0); // relative velocity vector
    dfloat3 n = wallData.normal * -1.0f; //invert collision direction since is from sphere to wall

    //vector center-> contact 
    dfloat3 rr = pos_c_i - pos_i;
    dfloat3 G = v_i - wall_speed;


    //collision constants
    const dfloat STIFFNESS_NORMAL = SPHERE_WALL_STIFFNESS_NORMAL_CONST * sqrt(abs(cr[0]));
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_WALL_STIFFNESS_TANGENTIAL_CONST * sqrt(cr[0]) * sqrt (abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_WALL_DAMPING_CONST * sqrt(m_i * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n + rr) - dot_product(G, n) * n;
    dfloat mag = vector_length(G_ct);
    dfloat3 t = (mag != 0) ? (G_ct / mag) : dfloat3{0.0, 0.0, 0.0}; // Tangential velocity vector


    //retrive and update tangential displacement
    int tang_index = calculateWallIndex(n); //wall can be directly computed
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, 0, true, step, G_ct, G, tang_index, n);

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, t, pc_i, tang_index, step
    );

    //sum the forces
    dfloat3 f_dirs = f_normal + f_tang;

    //calculate moments
    dfloat3 m_dirs = cross_product(rr,f_dirs);

    //save date in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs);
}

__device__
dfloat ellipsoidWallCollisionDistance( ParticleCenter* pc_i, Wall wallData,dfloat3 contactPoint2[1], dfloat radius[1], unsigned int step){
    //contruct rotation matrix
    dfloat R[3][3];
    dfloat dist, error;
    dfloat3 new_sphere_center1, new_sphere_center2;
    dfloat3 closest_point1, closest_point2;

    dfloat a = vector_length(pc_i->getSemiAxis1()-pc_i->getPos());
    dfloat b = vector_length(pc_i->getSemiAxis2()-pc_i->getPos());
    dfloat c = vector_length(pc_i->getSemiAxis3()-pc_i->getPos());

    rotationMatrixFromVectors((pc_i->getSemiAxis1() - pc_i->getPos())/a,(pc_i->getSemiAxis2() - pc_i->getPos())/b,(pc_i->getSemiAxis3() - pc_i->getPos())/c,R);


    //projection of center into wall
    dfloat3 proj = planeProjection(pc_i->getPos(),wallData.normal,wallData.distance);
    dfloat3 dir = pc_i->getPos() - proj;
    dfloat3 t = ellipsoid_intersection(pc_i,R,proj,dir,dfloat3(0,0,0));
    dfloat3 inter1 = proj + t.x*dir;
    dfloat3 inter2 = proj + t.y*dir;


    if (dot_product(inter1,wallData.normal) < dot_product(inter2,wallData.normal)){
        closest_point2 = inter1;
    }else{
        closest_point2 = inter2;
    }    
    
    dfloat r = 3; //TODO: FIND A BETTER WAY TI DETERMINE IT

    //compute normal vector at intersection
    dfloat3 normal2 = ellipsoid_normal(pc_i,R,closest_point2,radius,dfloat3(0,0,0));

    //Compute the centers of the spheres in the opposite direction of the normals
    dfloat3 sphere_center2 = closest_point2 - r * normal2;

    //Iteration loop
    dfloat max_iters = 20;
    dfloat tolerance = 1e-3;

    for(int i = 0; i< max_iters;i++){
        proj = planeProjection(sphere_center2,wallData.normal,wallData.distance);
        dir = sphere_center2 - proj;
        t = ellipsoid_intersection(pc_i,R,proj,dir,dfloat3(0,0,0));

        inter1 = proj + t.x*dir;
        inter2 = proj + t.y*dir;

        if (dot_product(inter1,wallData.normal) < dot_product(inter2,wallData.normal)){
            closest_point2 = inter1;
        }else{
            closest_point2 = inter2;
        }    

        normal2 = ellipsoid_normal(pc_i,R,closest_point2,radius,dfloat3(0,0,0));        
        new_sphere_center2 = closest_point2 - r * normal2;

        error = vector_length(new_sphere_center2 - sphere_center2);
        if (error < tolerance ){
            break;      
        }else{
            //update values
            sphere_center2 = new_sphere_center2;
        }
    }

    contactPoint2[0] = closest_point2;
    dist = vector_length(sphere_center2 - proj) - r;
    return dist;

}

// ****************************************************************************
// ************************   PARTICLE COLLISION   ****************************
// ****************************************************************************

// -------------------------- SPHERE COLLISION --------------------------------
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
        abs(pos_i.x - pos_j.x) > ((NX-1) / 2.0) ? 
        (pos_i.x < pos_j.x ?
            (pos_i.x + (NX-1) - pos_j.x)
            : 
            (pos_i.x - (NX-1) - pos_j.x)
        )
        : pos_i.x - pos_j.x
        #endif //BC_X_PERIODIC
        , 
        #ifdef BC_Y_WALL
           (pos_i.y - pos_j.y)
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
        abs(pos_i.y - pos_j.y) > ((NY-1) / 2.0) ? 
        (pos_i.y < pos_j.y ?
            (pos_i.y + (NY-1) - pos_j.y)
            : 
            (pos_i.y - (NY-1) - pos_j.y)
        )
        : pos_i.y - pos_j.y
        #endif //BC_Y_PERIODIC
        , 
        #ifdef BC_Z_WALL
            pos_i.z - pos_j.z
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            abs(pos_i.z - pos_j.z) > ((NZ-1) / 2.0) ? 
            (pos_i.z < pos_j.z ?
                (pos_i.z + (NZ-1) - pos_j.z)
                : 
                (pos_i.z - (NZ-1) - pos_j.z)
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

// ------------------------- CAPSULE COLLISIONS -------------------------------
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
        abs(pos_i.x - pos_j.x) > ((NX-1) / 2.0) ? 
        (pos_i.x < pos_j.x ?
            (pos_i.x + (NX-1) - pos_j.x)
            : 
            (pos_i.x - (NX-1) - pos_j.x)
        )
        : pos_i.x - pos_j.x
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL
            pos_i.y - pos_j.y
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
        abs(pos_i.y - pos_j.y) > ((NY-1) / 2.0) ? 
        (pos_i.y < pos_j.y ?
            (pos_i.y + (NY-1) - pos_j.y)
            : 
            (pos_i.y - (NY-1) - pos_j.y)
        )
        : pos_i.y - pos_j.y
        #endif //BC_Y_PERIODIC
        ,
        #ifdef BC_Z_WALL
            pos_i.z - pos_j.z
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            abs(pos_i.z - pos_j.z) > ((NZ-1) / 2.0) ? 
            (pos_i.z < pos_j.z ?
                (pos_i.z + (NZ-1) - pos_j.z)
                : 
                (pos_i.z - (NZ-1) - pos_j.z)
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

// ------------------------ ELLIPSOID COLLISIONS ------------------------------
__device__
void ellipsoidEllipsoidCollision(unsigned int column, unsigned int row, ParticleCenter*  pc_i, ParticleCenter*  pc_j,dfloat3 closestOnA[1], dfloat3 closestOnB[1], dfloat dist, dfloat cr1[1], dfloat cr2[1], dfloat3 translation, int step){
    // Particle i info (column)
    const dfloat3 pos_i = closestOnA[0];
    const dfloat3 pos_c_i = pc_i->getPos();
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
   
    // Particle j info (row)
    const dfloat3 pos_j = closestOnB[0] + translation;
    const dfloat3 pos_c_j = pc_j->getPos() + translation;
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    const dfloat3 diff_pos = dfloat3(
        #ifdef BC_X_WALL
            pos_i.x - pos_j.x
        #endif //BC_X_WALL
        #ifdef BC_X_PERIODIC 
        abs(pos_i.x - pos_j.x) > ((NX-1) / 2.0) ? 
        (pos_i.x < pos_j.x ?
            (pos_i.x + (NX-1) - pos_j.x)
            : 
            (pos_i.x - (NX-1) - pos_j.x)
        )
        : pos_i.x - pos_j.x
        #endif //BC_X_PERIODIC
        ,
        #ifdef BC_Y_WALL
            pos_i.y - pos_j.y
        #endif //BC_Y_WALL
        #ifdef BC_Y_PERIODIC
        abs(pos_i.y - pos_j.y) > ((NY-1) / 2.0) ? 
        (pos_i.y < pos_j.y ?
            (pos_i.y + (NY-1) - pos_j.y)
            : 
            (pos_i.y - (NY-1) - pos_j.y)
        )
        : pos_i.y - pos_j.y
        #endif //BC_Y_PERIODIC
        ,
        #ifdef BC_Z_WALL
            pos_i.z - pos_j.z
        #endif //BC_Z_WALL
        #ifdef BC_Z_PERIODIC
            abs(pos_i.z - pos_j.z) > ((NZ-1) / 2.0) ? 
            (pos_i.z < pos_j.z ?
                (pos_i.z + (NZ-1) - pos_j.z)
                : 
                (pos_i.z - (NZ-1) - pos_j.z)
            )
            : pos_i.z - pos_j.z
        #endif //BC_Z_PERIODIC
    );

    const dfloat mag_dist = abs(dist);

    //normal collision vector
    const dfloat3 n = dfloat3(diff_pos.x/mag_dist,diff_pos.y/mag_dist,diff_pos.z/mag_dist);

    //normal deformation
    dfloat displacement = -dist;

    //vector center-> contact 
    dfloat3 rri = pos_i - pos_c_i;
    dfloat3 rrj = pos_j - pos_c_j;

    // relative velocity vector
    dfloat3 G = v_i-v_j;

    //HERTZ CONTACT THEORY

    //dfloat effective_radius = 1.0/((r_i +r_j)/(r_i*r_j));
    dfloat effective_radius = 1.0/((cr1[0] +cr2[0])/(cr1[0]*cr2[0]));
    dfloat effective_mass = 1.0/((m_i +m_j)/(m_i*m_j));

    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt (abs(displacement));
    const dfloat damping_const = (- 2.0 * log(PP_REST_COEF)  / (sqrt(M_PI*M_PI + log(PP_REST_COEF)*log(PP_REST_COEF)))); //TODO FIND A WAY TO PROCESS IN COMPILE TIME
    const dfloat DAMPING_NORMAL = damping_const * sqrt (effective_mass * STIFFNESS_NORMAL );
    const dfloat DAMPING_TANGENTIAL = damping_const * sqrt (effective_mass * STIFFNESS_TANGENTIAL);


    //normal force
    dfloat f_kn = -STIFFNESS_NORMAL * sqrt(abs(displacement*displacement*displacement));
    dfloat3 f_normal = f_kn * n - DAMPING_NORMAL *  dot_product(G,n) * n * POW_FUNCTION(abs(displacement),0.25);
    dfloat f_n = vector_length(f_normal);


    //tangential force
    dfloat3 G_ct = G + r_i * cross_product(w_i,n+rri) - dot_product(G,n)*n;
    dfloat mag = vector_length(G_ct);


        //calculate tangential vector
    dfloat3 t;
    if (mag != 0){
        //tangential vector
        t = G_ct / mag;
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
            tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,G_ct,step);
        }
    }


    dfloat3 f_tang;
    f_tang.x = - STIFFNESS_TANGENTIAL * tang_disp.x - DAMPING_TANGENTIAL * G_ct.x* POW_FUNCTION(abs(tang_disp.x) ,0.25);
    f_tang.y = - STIFFNESS_TANGENTIAL * tang_disp.y - DAMPING_TANGENTIAL * G_ct.y* POW_FUNCTION(abs(tang_disp.y) ,0.25);
    f_tang.z = - STIFFNESS_TANGENTIAL * tang_disp.z - DAMPING_TANGENTIAL * G_ct.z* POW_FUNCTION(abs(tang_disp.z) ,0.25);

    mag = vector_length(f_tang);

    //calculate if will slip
    if(  mag > PP_FRICTION_COEF * abs(f_n) ){
        tang_disp = updateTangentialDisplacement(pc_i->getCollision(),tang_index,-G_ct,step);
        f_tang = - PP_FRICTION_COEF * f_n * t;
    }

    //FINAL FORCE RESULTS

    // Force in each direction
    dfloat3 f_dirs = f_normal + f_tang;
    //Torque in each direction
    dfloat3 m_dirs_i = cross_product(rri, f_dirs);
    dfloat3 m_dirs_j = cross_product(rrj, -f_dirs);
    
    // Force positive in particle i (column)
    atomicAdd(&(pc_i->getFXatomic()), f_dirs.x);
    atomicAdd(&(pc_i->getFYatomic()), f_dirs.y);
    atomicAdd(&(pc_i->getFZatomic()), f_dirs.z);

    atomicAdd(&(pc_i->getMXatomic()), m_dirs_i.x);
    atomicAdd(&(pc_i->getMYatomic()), m_dirs_i.y);
    atomicAdd(&(pc_i->getMZatomic()), m_dirs_i.z);

    // Force negative in particle j (row)
    atomicAdd(&(pc_j->getFXatomic()), -f_dirs.x);
    atomicAdd(&(pc_j->getFYatomic()), -f_dirs.y);
    atomicAdd(&(pc_j->getFZatomic()), -f_dirs.z);

    atomicAdd(&(pc_j->getMXatomic()), m_dirs_j.x); //normal vector takes care of negative sign
    atomicAdd(&(pc_j->getMYatomic()), m_dirs_j.y);
    atomicAdd(&(pc_j->getMZatomic()), m_dirs_j.z); 

}


// ****************************************************************************
// ******************   AUXILIARY COLLISION FUNCTIONS  ************************
// ****************************************************************************
__device__
dfloat3 ellipsoid_intersection(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 line_origin, dfloat3 line_dir,dfloat3 translation){
    dfloat3 p0, p0_rotated, d_rotated;
    dfloat A, B, C;
    dfloat DELTA;
    dfloat3 t;

    dfloat3 center = pc_i->getPos() + translation;

    dfloat a_axis = vector_length(pc_i->getSemiAxis1()-pc_i->getPos());
    dfloat b_axis = vector_length(pc_i->getSemiAxis2()-pc_i->getPos());
    dfloat c_axis = vector_length(pc_i->getSemiAxis3()-pc_i->getPos());

    p0 = line_origin - center; // Line origin relative to the ellipsoid center


    // Apply rotation to the line origin
    p0_rotated.x = R[0][0] * p0.x + R[0][1] * p0.y + R[0][2] * p0.z;
    p0_rotated.y = R[1][0] * p0.x + R[1][1] * p0.y + R[1][2] * p0.z;
    p0_rotated.z = R[2][0] * p0.x + R[2][1] * p0.y + R[2][2] * p0.z;

    // Transform the line direction into the ellipsoid's coordinate system
    d_rotated.x = R[0][0] * line_dir.x + R[0][1] * line_dir.y + R[0][2] * line_dir.z;
    d_rotated.y = R[1][0] * line_dir.x + R[1][1] * line_dir.y + R[1][2] * line_dir.z;
    d_rotated.z = R[2][0] * line_dir.x + R[2][1] * line_dir.y + R[2][2] * line_dir.z;



    // Ellipsoid equation coefficients (in rotated space)
    A = (d_rotated.x / a_axis) * (d_rotated.x / a_axis) + (d_rotated.y / b_axis) * (d_rotated.y / b_axis) + (d_rotated.z / c_axis) * (d_rotated.z / c_axis);
    B = 2 * ((p0_rotated.x * d_rotated.x) / (a_axis * a_axis) + (p0_rotated.y * d_rotated.y) / (b_axis * b_axis) + (p0_rotated.z * d_rotated.z) / (c_axis * c_axis));
    C = (p0_rotated.x / a_axis) * (p0_rotated.x / a_axis) + (p0_rotated.y / b_axis) * (p0_rotated.y / b_axis) + (p0_rotated.z / c_axis) * (p0_rotated.z / c_axis) - 1;

    DELTA = B*B - 4*A*C;

    t = dfloat3((-B + sqrtf(DELTA)) / (2.0 * A),(-B - sqrtf(DELTA)) / (2.0 * A),0);

    return t;
}
__device__
dfloat3 ellipsoid_normal(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 point, dfloat radius[1],dfloat3 translation){
    dfloat3 local_point;
    dfloat3 grad_local;
    dfloat3 normal;
    dfloat norm;


    dfloat3 center = pc_i->getPos() + translation;
    
    dfloat a_axis = vector_length(pc_i->getSemiAxis1()-pc_i->getPos());
    dfloat b_axis = vector_length(pc_i->getSemiAxis2()-pc_i->getPos());
    dfloat c_axis = vector_length(pc_i->getSemiAxis3()-pc_i->getPos());

    // Transform the point into the ellipsoid's local coordinates
    local_point.x = (R[0][0] * (point.x - center.x) + R[0][1] * (point.y - center.y) + R[0][2] * (point.z - center.z));
    local_point.y = (R[1][0] * (point.x - center.x) + R[1][1] * (point.y - center.y) + R[1][2] * (point.z - center.z));
    local_point.z = (R[2][0] * (point.x - center.x) + R[2][1] * (point.y - center.y) + R[2][2] * (point.z - center.z));

    dfloat a4 = a_axis*a_axis*a_axis*a_axis;
    dfloat b4 = b_axis*b_axis*b_axis*b_axis;
    dfloat c4 = c_axis*c_axis*c_axis*c_axis;


    radius[0] = (a_axis*b_axis*c_axis*(a4*b4*local_point.z*local_point.z + a4*c4*local_point.y*local_point.y+ b4*c4*local_point.x*local_point.x))/(a4*b4*c4);

    // Compute the gradient in local coordinates
    grad_local.x = 2 * local_point.x / (a_axis * a_axis);
    grad_local.y = 2 * local_point.y / (b_axis * b_axis);
    grad_local.z = 2 * local_point.z / (c_axis * c_axis);

    // Transform the gradient back to global coordinates
    normal.x = R[0][0] * grad_local.x + R[1][0] * grad_local.y + R[2][0] * grad_local.z;
    normal.y = R[0][1] * grad_local.x + R[1][1] * grad_local.y + R[2][1] * grad_local.z;
    normal.z = R[0][2] * grad_local.x + R[1][2] * grad_local.y + R[2][2] * grad_local.z;

    // Normalize the normal vector
    norm = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (norm > 0) { // Avoid division by zero
        normal.x /= norm;
        normal.y /= norm;
        normal.z /= norm;
    }


    return normal;
}
__device__
dfloat ellipsoidEllipsoidCollisionDistance( ParticleCenter* pc_i, ParticleCenter* pc_j,dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr1[1], dfloat cr2[1], dfloat3 translation, unsigned int step){
    dfloat R1[3][3];
    dfloat R2[3][3];
    dfloat dist, error;
    dfloat3 new_sphere_center1, new_sphere_center2;
    dfloat3 closest_point1, closest_point2;

    //obtain semi-axis values
    dfloat a1 = vector_length(pc_i->getSemiAxis1()-pc_i->getPos());
    dfloat b1 = vector_length(pc_i->getSemiAxis2()-pc_i->getPos());
    dfloat c1 = vector_length(pc_i->getSemiAxis3()-pc_i->getPos());
    
    dfloat a2 = vector_length(pc_j->getSemiAxis1()-pc_j->getPos());
    dfloat b2 = vector_length(pc_j->getSemiAxis2()-pc_j->getPos());
    dfloat c2 = vector_length(pc_j->getSemiAxis3()-pc_j->getPos());


    //obtain rotation matrix
    rotationMatrixFromVectors((pc_i->getSemiAxis1() - pc_i->getPos())/a1,(pc_i->getSemiAxis2() - pc_i->getPos())/b1,(pc_i->getSemiAxis3() - pc_i->getPos())/c1,R1);
    rotationMatrixFromVectors((pc_j->getSemiAxis1() - pc_j->getPos())/a2,(pc_j->getSemiAxis2() - pc_j->getPos())/b2,(pc_j->getSemiAxis3() - pc_j->getPos())/c2,R2);

    dfloat3 dir = (pc_j->getPos() + translation) - pc_i->getPos();

    dfloat3 t1 = ellipsoid_intersection(pc_i,R1,pc_i->getPos(),dir,dfloat3(0,0,0));
    dfloat3 t2 = ellipsoid_intersection(pc_j,R2,pc_i->getPos(),dir,translation);

    computeContactPoints(pc_i->getPos(), dir, t1,  t2, &closest_point1, &closest_point2);

    dfloat r = 3; //TODO FIND A BETTER WAY TO GET THE BEST RADIUS FOR THIS DETECTION


    //compute normal vector at intersection
    dfloat3 normal1 = ellipsoid_normal(pc_i,R1,closest_point1,cr1,dfloat3(0,0,0));
    dfloat3 normal2 = ellipsoid_normal(pc_j,R2,closest_point2,cr2,translation);
    dfloat3 sphere_center1 = closest_point1 - r * normal1;
    dfloat3 sphere_center2 = closest_point2 - r * normal2;


    //Iteration loop
    dfloat max_iters = 20;
    dfloat tolerance = 1e-3;
    for(int i = 0; i< max_iters;i++){
         dir = sphere_center2 - sphere_center1;
         
        t1 = ellipsoid_intersection(pc_i,R1,sphere_center1,dir,dfloat3(0,0,0));
        t2 = ellipsoid_intersection(pc_j,R2,sphere_center1,dir,translation);

        computeContactPoints(sphere_center1, dir, t1, t2, &closest_point1, &closest_point2);

        normal1 = ellipsoid_normal(pc_i,R1,closest_point1,cr1,dfloat3(0,0,0));
        normal2 = ellipsoid_normal(pc_j,R2,closest_point2,cr2,translation);

        new_sphere_center1 = closest_point1 - r * normal1;
        new_sphere_center2 = closest_point2 - r * normal2;

        error = vector_length(new_sphere_center2 - sphere_center2) + vector_length(new_sphere_center1 - sphere_center1);
        if (error < tolerance ){
            break;      
        }else{
            //update values
            sphere_center1 = new_sphere_center1;
            sphere_center2 = new_sphere_center2;
        }
    }

    contactPoint1[0] = closest_point1;
    contactPoint2[0] = closest_point2;
    dist = vector_length(sphere_center2 - sphere_center1) - 2*r;
    return dist;

}
__device__
void computeContactPoints(dfloat3 pos_i, dfloat3 dir, dfloat3 t1, dfloat3 t2, dfloat3 contactPoint1[1], dfloat3 contactPoint2[1])
{


    dfloat3 inter11 = pos_i + t1.x * dir;
    dfloat3 inter12 = pos_i + t1.y * dir;
    dfloat3 inter21 = pos_i + t2.x * dir;
    dfloat3 inter22 = pos_i + t2.y * dir;

    // compute distances
    dfloat distances[2][2];
    distances[0][0] = vector_length(inter11 - inter21);
    distances[0][1] = vector_length(inter11 - inter22);
    distances[1][0] = vector_length(inter12 - inter21);
    distances[1][1] = vector_length(inter12 - inter22);

    // find minimum distances
    dfloat min_dist = 1e37;
    int i_min = 0;
    int j_min = 0;
    if (distances[0][0] < min_dist)
    {
        min_dist = distances[0][0];
        i_min = 0;
        j_min = 0;
    }
    if (distances[0][1] < min_dist)
    {
        min_dist = distances[0][1];
        i_min = 0;
        j_min = 1;
    }
    if (distances[1][0] < min_dist)
    {
        min_dist = distances[1][0];
        i_min = 1;
        j_min = 0;
    }
    if (distances[1][1] < min_dist)
    {
        min_dist = distances[1][1];
        i_min = 1;
        j_min = 1;
    }

    switch (i_min * 2 + j_min)
    {
    case 0: // i_min = 0, j_min = 0
        contactPoint1[0] = inter11;
        contactPoint2[0] = inter21;
        break;
    case 1: // i_min = 0, j_min = 1
        contactPoint1[0] = inter11;
        contactPoint2[0] = inter22;
        break;
    case 2: // i_min = 1, j_min = 0
        contactPoint1[0] = inter12;
        contactPoint2[0] = inter21;
        break;
    case 3: // i_min = 1, j_min = 1
        contactPoint1[0] = inter12;
        contactPoint2[0] = inter22;
        break;
    default:
        break;
    }
}


#endif //PARTICLE_MODEL