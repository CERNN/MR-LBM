

//functions to determine if the particle will collide
#include "collisionDetection.cuh"

#ifdef PARTICLE_MODEL

// ------------------------------------------------------------------------ 
// -------------------- COLLISION HANDLER --------- -----------------------
// ------------------------------------------------------------------------ 

//collision
__global__
void gpuParticlesCollisionHandler(ParticleShape *shape, ParticleCenter *pArray, unsigned int step){
    /* Maps a 1D array to a Floyd triangle, where the last row is for checking
    collision against the wall and the other ones to check collision between 
    particles, with index given by row/column. Example for 7 particles:

    FLOYD TRIANGLE
        c0  c1  c2  c3  c4  c5  c6
    r0  0
    r1  1   2
    r2  3   4   5
    r3  6   7   8   9
    r4  10  11  12  13  14
    r5  15  16  17  18  19  20
    r6  21  22  23  24  25  26  27

    Index 7 is in r3, c1. It will compare p[1] (particle in index 1), from column,
    with p[4], from row (this is because for all rows one is added to its index)

    Index 0 will compare p[0] (column) and p[1] (row)
    Index 13 will compare p[3] (column) and p[5] (row)
    Index 20 will compare p[5] (column) and p[6] (row)

    For the last column, the particles check collision against the wall.
    Index 21 will check p[0] (column) collision against the wall
    Index 27 will check p[6] (column) collision against the wall
    Index 24 will check p[3] (column) collision against the wall

    FROM INDEX TO ROW/COLUMN
    Starting column/row from 1, the n'th row always ends (n)*(n+1)/2+1. So:

    k = (n)*(n+1)/2+1
    n^2 + n - (2k+1) = 0

    (with k=particle index)
    n_row = ceil((-1 + Sqrt(1 + 8(k+1))) / 2)
    n_column = k - n_row * (n_row - 1) / 2
    */
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx > TOTAL_PCOLLISION_IBM_THREADS)
        return;
    
    const unsigned int row = ceil((-1.0+sqrt((dfloat)1+8*(idx+1)))/2);
    const unsigned int column = idx - ((row-1)*row)/2;

    ParticleCenter* pc_i = &pArray[column];
    ParticleShape* shape_i = &shape[column];
   
    //collision against walls
    if(row == NUM_PARTICLES){
        if(!pc_i->getMovable())
            return;
        checkCollisionWalls(shape_i,pc_i,step);
    }else{    //Collision between particles
        ParticleCenter* pc_j = &pArray[row]; 
        ParticleShape* shape_j = &shape[row];
        if(!pc_i->getMovable() && !pc_j->getMovable())
            return;
        checkCollisionBetweenParticles(column,row,shape_i,shape_j,pc_i,pc_j,step);
    }
}

__device__
void checkCollisionWalls(ParticleShape *shape, ParticleCenter* pc_i, unsigned int step){
    switch (*shape) {
        case SPHERE:
            checkCollisionWallsSphere(pc_i,step);
            break;
        case CAPSULE:
            checkCollisionWallsCapsule(pc_i,step);
            break;
        case ELLIPSOID:
            checkCollisionWallsElipsoid(pc_i,step);
            break;
        default:
            // Handle unknown particle types
            break;
    }  
}

__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i, unsigned int step) {
    const dfloat3 pos_i = pc_i->getPos();
    const dfloat radius = pc_i->getRadius();

    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0);
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), (NX - 1));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0);
    walls[wallCount++] = wall(dfloat3(0, -1, 0), (NY - 1));
    #endif

    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0);
    walls[wallCount++] = wall(dfloat3(0, 0, -1), (NZ_TOTAL - 1));
    #endif

    for (int i = 0; i < wallCount; ++i) {
        dfloat distanceWall;
        
        // Check the direction of the wall normal to use the correct distance calculation
        // The positive normal vectors point inward from a boundary at 0, while
        // the negative normal vectors point inward from a boundary at N-1.
        if (walls[i].normal.x + walls[i].normal.y + walls[i].normal.z > 0) {
            distanceWall = dot_product(pos_i, walls[i].normal) - walls[i].distance;
        } else {
            distanceWall = walls[i].distance + dot_product(pos_i, walls[i].normal);
        }

        if (distanceWall < radius) {
            sphereWallCollision({pc_i, walls[i], radius - distanceWall, step});
        }
    }
}

__device__
void checkCollisionWallsCapsule(ParticleCenter* pc_i, unsigned int step) {
    const dfloat halfLength = vector_length(pc_i->getSemiAxis1());
    const dfloat radius = pc_i->getRadius();
    const dfloat3 endpoint1 = pc_i->getSemiAxis1();
    const dfloat3 endpoint2 = pc_i->getSemiAxis2();

    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0);
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), (NX - 1));
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0);
    walls[wallCount++] = wall(dfloat3(0, -1, 0), (NY - 1));
    #endif
    
    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0);
    walls[wallCount++] = wall(dfloat3(0, 0, -1), (NZ - 1));
    #endif

    for (int i = 0; i < wallCount; ++i) {
        //Need to handle the two different distance calculations based on the wall
        dfloat distanceWall1;
        dfloat distanceWall2;
        
        // We need to determine if it's an inward- or outward-facing wall.
        // A simple way is to check the normal vector.
        if (walls[i].normal.x + walls[i].normal.y + walls[i].normal.z > 0) { // Outward facing
            distanceWall1 = dot_product(endpoint1, walls[i].normal) - walls[i].distance;
            distanceWall2 = dot_product(endpoint2, walls[i].normal) - walls[i].distance;
        } else { // Inward facing
            distanceWall1 = walls[i].distance + dot_product(endpoint1, walls[i].normal);
            distanceWall2 = walls[i].distance + dot_product(endpoint2, walls[i].normal);
        }

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap({pc_i, walls[i], radius - distanceWall1, step, endpoint1});
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap({pc_i, walls[i], radius - distanceWall2, step, endpoint2});
        }
    }
}

__device__
void checkCollisionWallsElipsoid(ParticleCenter* pc_i, unsigned int step) {
    const int maxWalls = 6;
    Wall walls[maxWalls];
    int wallCount = 0;

    #ifdef BC_X_WALL
    walls[wallCount++] = wall(dfloat3(1, 0, 0), 0);
    walls[wallCount++] = wall(dfloat3(-1, 0, 0), NX - 1);
    #endif

    #ifdef BC_Y_WALL
    walls[wallCount++] = wall(dfloat3(0, 1, 0), 0);
    walls[wallCount++] = wall(dfloat3(0, -1, 0), NY - 1);
    #endif
    
    #ifdef BC_Z_WALL
    walls[wallCount++] = wall(dfloat3(0, 0, 1), 0);
    walls[wallCount++] = wall(dfloat3(0, 0, -1), NZ - 1);
    #endif

    // Loop through all defined walls and check for collision
    for (int i = 0; i < wallCount; ++i) {
        dfloat distanceWall = 0;
        dfloat3 contactPoint2[1];
        dfloat cr[1];

        distanceWall = ellipsoidWallCollisionDistance(pc_i, walls[i], contactPoint2, cr, step);
        if (distanceWall < 0) {
            ellipsoidWallCollision({pc_i, walls[i], -distanceWall, step, contactPoint2[0]}, cr);
        }
    }
}

// ------------------------------------------------------------------------
// -------------------- COLLISION BETWEEN PARTICLES -----------------------
// ------------------------------------------------------------------------ 

__device__
void checkCollisionBetweenParticles( unsigned int column,unsigned int row,ParticleShape *shape_i,ParticleShape *shape_j,ParticleCenter* pc_i,ParticleCenter* pc_j,int step){

    switch (*shape_i) {
        case SPHERE:
            switch (*shape_j) {
            case SPHERE:
                if(sphereSphereGap( pc_i, pc_j)<0){
                    sphereSphereCollision(column,row, pc_i, pc_j,step);
                }
                break;
            case CAPSULE:
                capsuleSphereCollisionCheck(column,row,shape_i,pc_i,pc_j,step);
                break;
            case ELLIPSOID:
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        case CAPSULE:
            switch (*shape_j) {
            case SPHERE:
                capsuleSphereCollisionCheck(column,row,shape_i,pc_i,pc_j,step);
                break;
            case CAPSULE:
                capsuleCapsuleCollisionCheck(column,row,pc_i,pc_j, step, pc_i->getSemiAxis1(),pc_i->getSemiAxis2(), pc_i->getRadius(),pc_j->getSemiAxis1(), pc_j->getSemiAxis2(), pc_j->getRadius());
            case ELLIPSOID:
                //collision capsule-ellipsoid
                break;
            default:
                break;
            }
            break;
        case ELLIPSOID:
            switch (*shape_j) {
            case SPHERE:
                //collision ellipsoid-sphere
                break;
            case CAPSULE:
                //collision ellipsoid-capsule
                break;
            case ELLIPSOID:
                ellipsoidEllipsoidCollisionCheck(column,row,pc_i,pc_j, step);
                break;
            default:
                // Handle unknown particle types
                break;
            }
            break;
        default:
            // Handle unknown particle types
            break;
    }
}

// ------------------------------------------------------------------------ 
// -------------------- INTER PARTICLE COLLISION CHECK---------------------
// ------------------------------------------------------------------------ 

__device__
void capsuleCapsuleCollisionCheck(    unsigned int column,    unsigned int row,ParticleCenter* pc_i, ParticleCenter* pc_j, int step, dfloat3 cylA1, dfloat3 cylA2, dfloat radiusA, dfloat3 cylB1, dfloat3 cylB2, dfloat radiusB) {
    dfloat3 closestOnA[1];
    dfloat3 closestOnB[1];

    if(segment_segment_closest_points_periodic(cylA1, cylA2, cylB1, cylB2, closestOnA, closestOnB) < radiusA + radiusB){
        capsuleCapsuleCollision(column, row,pc_i,pc_j,closestOnA,closestOnB,step);
    }

    return;
}

__device__
void capsuleSphereCollisionCheck(unsigned int column,unsigned int row,  ParticleShape *shape, ParticleCenter* pc_i, ParticleCenter* pc_j, int step){

    dfloat3 closestOnAB[1];
    
    if(*shape == SPHERE){
        dfloat3 pos_i = pc_i->getPos();
        if(point_to_segment_distance_periodic(pc_i->getPos(), pc_j->getSemiAxis1(), pc_j->getSemiAxis2(),closestOnAB) < pc_i->getRadius() + pc_j->getRadius())
            capsuleCapsuleCollision(column,row,pc_i,pc_j,&pos_i,closestOnAB,step);
    }else{
        dfloat3 pos_j = pc_j->getPos();
        if(point_to_segment_distance_periodic(pc_j->getPos(), pc_i->getSemiAxis1(), pc_i->getSemiAxis2(),closestOnAB) < pc_i->getRadius() + pc_j->getRadius())
            capsuleCapsuleCollision(column,row,pc_i,pc_j,&pos_j,closestOnAB,step);
    }
    

    return;
}

__device__
void ellipsoidEllipsoidCollisionCheck(unsigned int column, unsigned int row, ParticleCenter* pc_i,ParticleCenter* pc_j, int step){
    dfloat3 closestOnA[1];
    dfloat3 closestOnB[1];
    dfloat cr1[1];
    dfloat cr2[1];

    dfloat minDist = 1E+37;  // Initialize to a large value
    dfloat3 bestClosestOnA;
    dfloat3 bestClosestOnB;
    dfloat bestcr1, bestcr2;
    int dx = 0, dy = 0, dz = 0;
    dfloat dist;
    dfloat3 translation, bestTranslation;

    // Loop over periodic offsets in x, y, and z if periodic boundary conditions are enabled
    #ifdef BC_X_PERIODIC
    for (dx = -1; dx <= 1; dx++) {
    #endif //BC_X_PERIODIC
        #ifdef BC_Y_PERIODIC
        for (dy = -1; dy <= 1; dy++) {
        #endif //BC_Y_PERIODIC
            #ifdef BC_Z_PERIODIC
            for (dz = -1; dz <= 1; dz++) {
            #endif //BC_Z_PERIODIC
                translation = dfloat3(dx * (NX-1), dy * (NY-1), dz * (NZ-1));
                dist = ellipsoidEllipsoidCollisionDistance(pc_i, pc_j,closestOnA,closestOnB,cr1, cr2,translation,step);
                // Update the minimum distance and store the closest point
                if (dist < minDist) {
                    minDist = dist;
                    bestClosestOnA = closestOnA[0];
                    bestClosestOnB = closestOnB[0];
                    bestcr1 = cr1[0];
                    bestcr2 = cr2[0];
                    bestTranslation = translation;
                }


            #ifdef BC_Z_PERIODIC
            } // End Z loop
            #endif //BC_Z_PERIODIC
        #ifdef BC_Y_PERIODIC
        } // End Y loop
        #endif //BC_Y_PERIODIC
    #ifdef BC_X_PERIODIC
    } // End X loop
    #endif //BC_X_PERIODIC

    // Store the closest point on the segment
    closestOnA[0] = bestClosestOnA;
    closestOnB[0] = bestClosestOnB-bestTranslation;
    cr1[0] = bestcr1;
    cr2[0] = bestcr2;
    dist = minDist;

    if(dist<0){
        ellipsoidEllipsoidCollision(column, row,pc_i,pc_j,closestOnA,closestOnB,dist,cr1, cr2,bestTranslation,step);
    }


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
    #endif //BC_X_PERIODIC
    #ifdef BC_Y_PERIODIC
        if(delta.y > NY / 2.0) delta.y -= NY;
        if(delta.y < -NY / 2.0) delta.y += NY;
    #endif //BC_Y_PERIODIC
    #ifdef BC_Z_PERIODIC
        if(delta.z > NZ / 2.0) delta.z -= NZ;
        if(delta.z < -NZ / 2.0) delta.z += NZ;
    #endif //BC_Z_PERIODIC

    dfloat dist = sqrtf(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    
    return dist - (r1 + r2);
}

#endif //PARTICLE_MODEL