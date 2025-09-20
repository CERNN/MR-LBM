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
    dfloat3 f_tang;
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
void sphereSphereCollision(const CollisionContext& ctx){
    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;
    int step = ctx.step;
    dfloat displacement = ctx.displacement;
    int partnerID = ctx.partnerID;
    // Particle i info
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    // Particle j info
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    // Use normal direction from context (set in wall.normal)
    const dfloat3 n = ctx.wall.normal;
    // Relative velocity
    dfloat3 G = v_i - v_j;

    // Hertz contact theory
    dfloat effective_radius = 1.0 / ((r_i + r_j) / (r_i * r_j));
    dfloat effective_mass = 1.0 / ((m_i + m_j) / (m_i * m_j));
    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n) - dot_product(G, n) * n;
    dfloat mag = vector_length(G_ct);
    dfloat3 t = (mag != 0) ? (G_ct / mag) : dfloat3{0.0, 0.0, 0.0}; //tangential velocity vector

    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, partnerID, false, step, G_ct, G, tang_index, dfloat3(0, 0, 0));

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PW_FRICTION_COEF, f_n, t, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = r_i * cross_product(n, f_tang);
    dfloat3 m_dirs_j = r_j * cross_product(n, f_tang);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, -f_dirs, m_dirs_i);
    accumulateForceAndTorque(pc_j,  f_dirs, m_dirs_j);
}

// ------------------------- CAPSULE COLLISIONS -------------------------------
__device__
void capsuleCapsuleCollision(const CollisionContext& ctx, dfloat3 closestOnA[1], dfloat3 closestOnB[1]) {
    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;
    int step = ctx.step;
    int partnerID = ctx.partnerID;
    dfloat displacement = ctx.displacement;
    // Contact points (capsule ends)
    dfloat3 pos_i = closestOnA[0]; // closest point on capsule i
    dfloat3 pos_c_i = pc_i->getPos();
    dfloat3 pos_j = closestOnB[0]; // closest point on capsule j
    dfloat3 pos_c_j = pc_j->getPos();
    
    // Particle info
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    // Use normal direction from context (set in wall.normal)
    const dfloat3 n = ctx.wall.normal;
    // Relative velocity
    dfloat3 G = v_i - v_j;

    // Hertz contact theory
    dfloat effective_radius = 1.0 / ((r_i + r_j) / (r_i * r_j));
    dfloat effective_mass = 1.0 / ((m_i + m_j) / (m_i * m_j));
    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));
    const dfloat DAMPING_NORMAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);

    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Relative tangential velocity
    dfloat3 G_ct = G + r_i * cross_product(w_i, n) + r_j * cross_product(w_j, n) - dot_product(G, n) * n;
    dfloat mag = vector_length(G_ct);
    dfloat3 t = (mag != 0) ? (G_ct / mag) : dfloat3{0.0, 0.0, 0.0};

    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, partnerID, false, step, G_ct, G, tang_index, dfloat3(0, 0, 0));

    // Compute tangential force
    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PP_FRICTION_COEF, f_n, t, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = cross_product((pos_i - pos_c_i) + (-n * r_i), -f_dirs);
    dfloat3 m_dirs_j = cross_product((pos_j - pos_c_j) + (n * r_j), f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, -f_dirs, m_dirs_i);
    accumulateForceAndTorque(pc_j,  f_dirs, m_dirs_j);
}

// ------------------------ ELLIPSOID COLLISIONS ------------------------------
__device__
void ellipsoidEllipsoidCollision(const CollisionContext& ctx,dfloat3 closestOnA[1], dfloat3 closestOnB[1],dfloat cr1[1], dfloat cr2[1], dfloat3 translation) {
    ParticleCenter* pc_i = ctx.pc_i;
    ParticleCenter* pc_j = ctx.pc_j;
    int step = ctx.step;
    int partnerID = ctx.partnerID;
    dfloat displacement = ctx.displacement;
    // Contact points
    dfloat3 pos_i = closestOnA[0]; // closest point on ellipsoid i
    dfloat3 pos_c_i = pc_i->getPos();
    dfloat3 pos_j = closestOnB[0] + translation; // closest point on ellipsoid j
    dfloat3 pos_c_j = pc_j->getPos() + translation;

    // Particle info
    const dfloat r_i = pc_i->getRadius();
    const dfloat m_i = pc_i->getVolume() * pc_i->getDensity();
    const dfloat3 v_i = pc_i->getVel();
    const dfloat3 w_i = pc_i->getW();
    const dfloat r_j = pc_j->getRadius();
    const dfloat m_j = pc_j->getVolume() * pc_j->getDensity();
    const dfloat3 v_j = pc_j->getVel();
    const dfloat3 w_j = pc_j->getW();

    // Use normal direction from context (set in wall.normal)
    const dfloat3 n = ctx.wall.normal;
    // Relative velocity
    dfloat3 G = v_i - v_j;

    // Hertz contact theory
    dfloat effective_radius = 1.0 / ((cr1[0] + cr2[0]) / (cr1[0] * cr2[0]));
    dfloat effective_mass = 1.0 / ((m_i + m_j) / (m_i * m_j));
    const dfloat STIFFNESS_NORMAL = SPHERE_SPHERE_STIFFNESS_NORMAL_CONST * sqrt(effective_radius);
    const dfloat STIFFNESS_TANGENTIAL = SPHERE_SPHERE_STIFFNESS_TANGENTIAL_CONST * sqrt(effective_radius) * sqrt(abs(displacement));

    const dfloat DAMPING_NORMAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_NORMAL);
    const dfloat DAMPING_TANGENTIAL = SPHERE_SPHERE_DAMPING_CONST * sqrt(effective_mass * STIFFNESS_TANGENTIAL);


    // Normal force
    dfloat3 f_normal = computeNormalForce(n, G, displacement, STIFFNESS_NORMAL, DAMPING_NORMAL);
    dfloat f_n = vector_length(f_normal);

    // Tangential force
    dfloat3 rri = pos_i - pos_c_i;
    dfloat3 rrj = pos_j - pos_c_j;
    dfloat3 G_ct = G + r_i * cross_product(w_i, n + rri) - dot_product(G, n) * n;
    dfloat mag = vector_length(G_ct);
    dfloat3 t = (mag != 0) ? (G_ct / mag) : dfloat3{0.0, 0.0, 0.0};

    int tang_index = -1;
    dfloat3 tang_disp = getOrUpdateTangentialDisplacement(pc_i, partnerID, false, step, G_ct, G, tang_index, dfloat3(0, 0, 0));

    dfloat3 f_tang = computeTangentialForce(
        tang_disp, G_ct, STIFFNESS_TANGENTIAL, DAMPING_TANGENTIAL,
        PP_FRICTION_COEF, f_n, t, pc_i, tang_index, step
    );

    // Final force results
    dfloat3 f_dirs = f_normal + f_tang;
    dfloat3 m_dirs_i = cross_product(rri, f_dirs);
    dfloat3 m_dirs_j = cross_product(rrj, -f_dirs);

    //Save data in the particle information
    accumulateForceAndTorque(pc_i, f_dirs, m_dirs_i);
    accumulateForceAndTorque(pc_j, -f_dirs, m_dirs_j);
}


// ****************************************************************************
// ******************   AUXILIARY COLLISION FUNCTIONS  ************************
// ****************************************************************************
__device__
dfloat3 ellipsoid_intersection(ParticleCenter* pc_i, dfloat R[3][3], dfloat3 line_origin, dfloat3 line_dir, dfloat3 translation) {
    dfloat3 p0, p0_rotated, d_rotated;
    dfloat A, B, C;
    dfloat DELTA;
    dfloat3 t;

    dfloat3 pos = pc_i->getPos();
    dfloat3 center = pos + translation;

    dfloat a_axis = vector_length(pc_i->getSemiAxis1() - pos);
    dfloat b_axis = vector_length(pc_i->getSemiAxis2() - pos);
    dfloat c_axis = vector_length(pc_i->getSemiAxis3() - pos);

    dfloat inva2 = 1.0f / (a_axis * a_axis);
    dfloat invb2 = 1.0f / (b_axis * b_axis);
    dfloat invc2 = 1.0f / (c_axis * c_axis);

    p0 = line_origin - center;

    // Apply rotation to the line origin
    p0_rotated.x = R[0][0] * p0.x + R[0][1] * p0.y + R[0][2] * p0.z;
    p0_rotated.y = R[1][0] * p0.x + R[1][1] * p0.y + R[1][2] * p0.z;
    p0_rotated.z = R[2][0] * p0.x + R[2][1] * p0.y + R[2][2] * p0.z;

    // Transform the line direction into the ellipsoid's coordinate system
    d_rotated.x = R[0][0] * line_dir.x + R[0][1] * line_dir.y + R[0][2] * line_dir.z;
    d_rotated.y = R[1][0] * line_dir.x + R[1][1] * line_dir.y + R[1][2] * line_dir.z;
    d_rotated.z = R[2][0] * line_dir.x + R[2][1] * line_dir.y + R[2][2] * line_dir.z;

    // Precompute scaled components
    dfloat px_inva2 = p0_rotated.x * inva2;
    dfloat py_invb2 = p0_rotated.y * invb2;
    dfloat pz_invc2 = p0_rotated.z * invc2;

    dfloat dx_inva2 = d_rotated.x * inva2;
    dfloat dy_invb2 = d_rotated.y * invb2;
    dfloat dz_invc2 = d_rotated.z * invc2;

    // Ellipsoid equation coefficients (in rotated space)
    A = d_rotated.x * dx_inva2 + d_rotated.y * dy_invb2 + d_rotated.z * dz_invc2;
    B = 2.0f * (p0_rotated.x * dx_inva2 + p0_rotated.y * dy_invb2 + p0_rotated.z * dz_invc2);
    C = p0_rotated.x * px_inva2 + p0_rotated.y * py_invb2 + p0_rotated.z * pz_invc2 - 1.0f;

    DELTA = B * B - 4.0f * A * C;

    dfloat sqrt_delta = sqrtf(DELTA);
    dfloat denom = 2.0f * A;
    t = dfloat3((-B + sqrt_delta) / denom, (-B - sqrt_delta) / denom, 0.0f);

    return t;
}

__device__
dfloat3 ellipsoid_normal(ParticleCenter* pc_i, dfloat R[3][3], dfloat3 point, dfloat radius[1], dfloat3 translation) {
    dfloat3 local_point;
    dfloat3 grad_local;
    dfloat3 normal;
    dfloat norm;

    dfloat3 pos = pc_i->getPos();
    dfloat3 center = pos + translation;

    dfloat a_axis = vector_length(pc_i->getSemiAxis1() - pos);
    dfloat b_axis = vector_length(pc_i->getSemiAxis2() - pos);
    dfloat c_axis = vector_length(pc_i->getSemiAxis3() - pos);

    dfloat a2 = a_axis * a_axis;
    dfloat b2 = b_axis * b_axis;
    dfloat c2 = c_axis * c_axis;

    dfloat inva2 = 1.0f / a2;
    dfloat invb2 = 1.0f / b2;
    dfloat invc2 = 1.0f / c2;

    dfloat a4 = a2 * a2;
    dfloat b4 = b2 * b2;
    dfloat c4 = c2 * c2;

    dfloat3 diff = point - center;

    // Transform the point into the ellipsoid's local coordinates
    local_point.x = R[0][0] * diff.x + R[0][1] * diff.y + R[0][2] * diff.z;
    local_point.y = R[1][0] * diff.x + R[1][1] * diff.y + R[1][2] * diff.z;
    local_point.z = R[2][0] * diff.x + R[2][1] * diff.y + R[2][2] * diff.z;

    dfloat x2 = local_point.x * local_point.x;
    dfloat y2 = local_point.y * local_point.y;
    dfloat z2 = local_point.z * local_point.z;

    dfloat a4b4 = a4 * b4;
    dfloat a4c4 = a4 * c4;
    dfloat b4c4 = b4 * c4;

    dfloat inner = a4b4 * z2 + a4c4 * y2 + b4c4 * x2;
    dfloat abc = a_axis * b_axis * c_axis;
    dfloat denom = a4 * b4 * c4;
    radius[0] = abc * inner / denom;

    // Compute the gradient in local coordinates
    grad_local.x = 2.0f * local_point.x * inva2;
    grad_local.y = 2.0f * local_point.y * invb2;
    grad_local.z = 2.0f * local_point.z * invc2;

    // Transform the gradient back to global coordinates
    normal.x = R[0][0] * grad_local.x + R[1][0] * grad_local.y + R[2][0] * grad_local.z;
    normal.y = R[0][1] * grad_local.x + R[1][1] * grad_local.y + R[2][1] * grad_local.z;
    normal.z = R[0][2] * grad_local.x + R[1][2] * grad_local.y + R[2][2] * grad_local.z;

    // Normalize the normal vector
    dfloat norm2 = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    if (norm2 > 0.0f) { // Avoid division by zero
        norm = sqrtf(norm2);
        dfloat inv_norm = 1.0f / norm;
        normal.x *= inv_norm;
        normal.y *= inv_norm;
        normal.z *= inv_norm;
    }

    return normal;
}

__device__
dfloat ellipsoidEllipsoidCollisionDistance(ParticleCenter* pc_i, ParticleCenter* pc_j, dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr1[1], dfloat cr2[1], dfloat3 translation, unsigned int step) {
    dfloat R1[3][3];
    dfloat R2[3][3];
    dfloat3 sphere_center1, sphere_center2;
    dfloat3 closest_point1, closest_point2;
    dfloat3 new_sphere_center1, new_sphere_center2;

    dfloat3 pos_i = pc_i->getPos();
    dfloat3 pos_j = pc_j->getPos();
    dfloat3 center_j = pos_j + translation;

    // Precompute semi-axis vectors and lengths
    dfloat3 vec_a1 = pc_i->getSemiAxis1() - pos_i;
    dfloat3 vec_b1 = pc_i->getSemiAxis2() - pos_i;
    dfloat3 vec_c1 = pc_i->getSemiAxis3() - pos_i;
    dfloat a1 = vector_length(vec_a1);
    dfloat b1 = vector_length(vec_b1);
    dfloat c1 = vector_length(vec_c1);

    dfloat3 vec_a2 = pc_j->getSemiAxis1() - pos_j;
    dfloat3 vec_b2 = pc_j->getSemiAxis2() - pos_j;
    dfloat3 vec_c2 = pc_j->getSemiAxis3() - pos_j;
    dfloat a2 = vector_length(vec_a2);
    dfloat b2 = vector_length(vec_b2);
    dfloat c2 = vector_length(vec_c2);

    // Compute unit vectors for rotation matrices
    dfloat3 unit_a1 = vec_a1 / a1;
    dfloat3 unit_b1 = vec_b1 / b1;
    dfloat3 unit_c1 = vec_c1 / c1;

    dfloat3 unit_a2 = vec_a2 / a2;
    dfloat3 unit_b2 = vec_b2 / b2;
    dfloat3 unit_c2 = vec_c2 / c2;

    // Obtain rotation matrices
    rotationMatrixFromVectors(unit_a1, unit_b1, unit_c1, R1);
    rotationMatrixFromVectors(unit_a2, unit_b2, unit_c2, R2);

    dfloat3 dir = center_j - pos_i;

    dfloat3 t1 = ellipsoid_intersection(pc_i, R1, pos_i, dir, dfloat3(0.0f, 0.0f, 0.0f));
    dfloat3 t2 = ellipsoid_intersection(pc_j, R2, pos_i, dir, translation);

    computeContactPoints(pos_i, dir, t1, t2, &closest_point1, &closest_point2);

    dfloat r = 3.0;  //TODO FIND A BETTER WAY TO GET THE BEST RADIUS FOR THIS DETECTION


    dfloat3 normal1 = ellipsoid_normal(pc_i, R1, closest_point1, cr1, dfloat3(0.0f, 0.0f, 0.0f));
    dfloat3 normal2 = ellipsoid_normal(pc_j, R2, closest_point2, cr2, translation);

        

    sphere_center1 = closest_point1 - r * normal1;
    sphere_center2 = closest_point2 - r * normal2;

    // Iteration loop with squared tolerance
    const int max_iters = 20;
    const dfloat tolerance_sq = 1e-6f;
    for (int i = 0; i < max_iters; ++i) {
        dir = sphere_center2 - sphere_center1;

        t1 = ellipsoid_intersection(pc_i, R1, sphere_center1, dir, dfloat3(0.0f, 0.0f, 0.0f));
        t2 = ellipsoid_intersection(pc_j, R2, sphere_center1, dir, translation);

        computeContactPoints(sphere_center1, dir, t1, t2, &closest_point1, &closest_point2);

        normal1 = ellipsoid_normal(pc_i, R1, closest_point1, cr1, dfloat3(0.0f, 0.0f, 0.0f));
        normal2 = ellipsoid_normal(pc_j, R2, closest_point2, cr2, translation);

        new_sphere_center1 = closest_point1 - r * normal1;
        new_sphere_center2 = closest_point2 - r * normal2;

        dfloat3 diff1 = new_sphere_center1 - sphere_center1;
        dfloat3 diff2 = new_sphere_center2 - sphere_center2;
        dfloat error_sq = (diff1.x * diff1.x + diff1.y * diff1.y + diff1.z * diff1.z) +
                          (diff2.x * diff2.x + diff2.y * diff2.y + diff2.z * diff2.z);

        if (error_sq < tolerance_sq) {
            break;
        }

        sphere_center1 = new_sphere_center1;
        sphere_center2 = new_sphere_center2;
    }

    contactPoint1[0] = closest_point1;
    contactPoint2[0] = closest_point2;

    dfloat3 center_diff = sphere_center2 - sphere_center1;
    return vector_length(center_diff) - 2.0f * r;
}
__device__
void computeContactPoints(dfloat3 pos_i, dfloat3 dir, dfloat3 t1, dfloat3 t2, dfloat3 contactPoint1[1], dfloat3 contactPoint2[1]) {
    // Calculate the squared distances between all four pairs of t values.
    // The vector_length(dir) is a constant scaling factor, so we can
    // omit it for the comparison and re-introduce it if needed later.
    dfloat dist_sq00 = (t1.x - t2.x) * (t1.x - t2.x);
    dfloat dist_sq01 = (t1.x - t2.y) * (t1.x - t2.y);
    dfloat dist_sq10 = (t1.y - t2.x) * (t1.y - t2.x);
    dfloat dist_sq11 = (t1.y - t2.y) * (t1.y - t2.y);

    // Find the minimum squared distance and its indices
    dfloat min_dist_sq = dist_sq00;
    int i_min = 0;
    int j_min = 0;

    if (dist_sq01 < min_dist_sq) {
        min_dist_sq = dist_sq01;
        i_min = 0;
        j_min = 1;
    }
    if (dist_sq10 < min_dist_sq) {
        min_dist_sq = dist_sq10;
        i_min = 1;
        j_min = 0;
    }
    if (dist_sq11 < min_dist_sq) {
        min_dist_sq = dist_sq11;
        i_min = 1;
        j_min = 1;
    }
    
    // Select the correct t values based on the minimum distance indices.
    dfloat ta = (i_min == 0) ? t1.x : t1.y;
    dfloat tb = (j_min == 0) ? t2.x : t2.y;

    // Compute the contact points just once.
    contactPoint1[0] = pos_i + ta * dir;
    contactPoint2[0] = pos_i + tb * dir;
}


#endif //PARTICLE_MODEL