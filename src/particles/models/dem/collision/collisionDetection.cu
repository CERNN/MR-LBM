

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
   
    #ifdef IBM_DEBUG
    printf("collision step %d x: %f \n",step,pc_i->pos.x);
    #endif
    //collision against walls
    if(row == NUM_PARTICLES){
        if(!pc_i->getMovable())
            return;
        //printf("checking collision with wall  \n");
        checkCollisionWalls(shape,pc_i,step);
    }else{    //Collision between particles
        ParticleCenter* pc_j = &pArray[row]; //&particles->getPCenterArray()[row];
        if(!pc_i->getMovable() && !pc_j->getMovable())
            return;
        //printf("checking collision with other particle  \n");
        checkCollisionBetweenParticles(column,row,shape,pc_i,pc_j,step);
    }
}

__device__
void checkCollisionWalls(ParticleShape *shape, ParticleCenter* pc_i, unsigned int step){
    //printf("checking particle shape for walls \n");
    switch (*shape) {
        case SPHERE:
            // printf("its a sphere \n");
            checkCollisionWallsSphere(pc_i,step);
            break;
        case CAPSULE:
           //printf("its a capsule \n");
            checkCollisionWallsCapsule(pc_i,step);
            break;
        case ELLIPSOID:
            //printf("its a ellipsoid \n");
            // checkCollisionWallsElipsoid(pc_i,step);
            break;
        default:
            // Handle unknown particle types
            break;
    }  
}

__device__
void checkCollisionWallsSphere(ParticleCenter* pc_i,unsigned int step){
    const dfloat3 pos_i = pc_i->getPos();
    // printf("checking collision with wall as sphere \n");
    // printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
    Wall wallData = wall(dfloat3( 0,0,0),0);
    dfloat distanceWall = 0;
    #ifdef BC_X_WALL
        wallData = wall(dfloat3( 1,0,0),0);
        distanceWall = dot_product(pos_i,wallData.normal) - wallData.distance;
        printf("distance to x = 0 is %f, radius = %f \n",distanceWall,pc_i->getRadius());
        if (distanceWall < pc_i->getRadius()){
            printf("colliding with x = 0 with a deformation of %f \n",pc_i->getRadius() - distanceWall);
            printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->getRadius() - distanceWall,step);

        }
        wallData = wall(dfloat3( -1,0,0),(NX - 1));
        //for this case the dot product will be always negative, while the first term will be always better, hence we have to invert and use + signal
        distanceWall = wallData.distance + dot_product(pos_i,wallData.normal);
        printf("distance to x = 1 is %f, radius = %f \n",distanceWall,pc_i->getRadius());
        if (distanceWall < pc_i->getRadius()){
            printf("colliding with x = 1 with a deformation of %f \n",pc_i->getRadius() - distanceWall);
            printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->getRadius() - distanceWall,step);
        }
    #endif
    #ifdef BC_Y_WALL
        wallData = wall(dfloat3(0,1,0),0);
        distanceWall = dot_product(pos_i,wallData.normal) - wallData.distance;
        //printf("distance to y = 0 is %f \n",distanceWall);
        if (distanceWall < pc_i->getRadius()){
            //printf("colliding with y = 0 with a deformation of %f \n",pc_i->getRadius() - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->getRadius() - distanceWall,step);
        }
        wallData = wall(dfloat3( 0,-1,0),(NY - 1));
        distanceWall = wallData.distance + dot_product(pos_i,wallData.normal);
        //printf("distance to y = 1 is %f \n",distanceWall);
        if (distanceWall < pc_i->getRadius()){
            //printf("colliding with y = 1 with a deformation of %f \n",pc_i->getRadius() - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->getRadius() - distanceWall,step);
        }
    #endif
    #ifdef BC_Z_WALL
        wallData = wall(dfloat3(0,0,1),0);
        distanceWall = dot_product(pos_i,wallData.normal) - wallData.distance;
        //printf("distance to z = 0 is %f \n",distanceWall);
        if (distanceWall < pc_i->getRadius()){
            //printf("colliding with z = 0  with a deformation of %f \n",pc_i->getRadius() -distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->getRadius() -  distanceWall,step);
        }
        wallData = wall(dfloat3( 0,0,-1),(NZ_TOTAL - 1));
        distanceWall = wallData.distance + dot_product(pos_i,wallData.normal); 
        //printf("distance to z = 1 is %f \n",distanceWall);
        if (distanceWall < pc_i->getRadius()){
            //printf("colliding with z = 1 with a deformation of %f \n",pc_i->getRadius() - distanceWall);
            //printf("particle position is x = %f y = %f z = %f \n ",pos_i.x,pos_i.y,pos_i.z);
            sphereWallCollision(pc_i,wallData,pc_i->getRadius() - distanceWall,step);
        }
    #endif

}

__device__
void checkCollisionWallsCapsule(ParticleCenter* pc_i,unsigned int step){
    const dfloat halfLength = vector_length(pc_i->getSemiAxis1());
    const dfloat radius = pc_i->getRadius();

    // Calculate capsule endpoints using the orientation vector
    dfloat3 endpoint1 = pc_i->getSemiAxis1();
    dfloat3 endpoint2 = pc_i->getSemiAxis2();

    Wall wallData = wall(dfloat3(0, 0, 0), 0);
    dfloat distanceWall1 = 0;
    dfloat distanceWall2 = 0;

    #ifdef BC_X_WALL
        wallData = wall(dfloat3(1, 0, 0), 0);
        distanceWall1 = dot_product(endpoint1, wallData.normal) - wallData.distance;
        distanceWall2 = dot_product(endpoint2, wallData.normal) - wallData.distance;

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

        wallData = wall(dfloat3(-1, 0, 0), (NX - 1));
        distanceWall1 = wallData.distance +  dot_product(endpoint1, wallData.normal);
        distanceWall2 = wallData.distance +  dot_product(endpoint2, wallData.normal);


        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

    #endif
    #ifdef BC_Y_WALL
        wallData = wall(dfloat3( 0,1,0),0);
        distanceWall1 = dot_product(endpoint1, wallData.normal) - wallData.distance;
        distanceWall2 = dot_product(endpoint2, wallData.normal) - wallData.distance;

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

        
        wallData = wall(dfloat3(0, 1, 0), (NY - 1));
        distanceWall1 = wallData.distance +  dot_product(endpoint1, wallData.normal);
        distanceWall2 = wallData.distance +  dot_product(endpoint2, wallData.normal);


        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {;
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }

    #endif
    #ifdef BC_Z_WALL
        wallData = wall(dfloat3( 0,0,1),0);
        distanceWall1 = dot_product(endpoint1, wallData.normal) - wallData.distance;
        distanceWall2 = dot_product(endpoint2, wallData.normal) - wallData.distance;

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }
        
        wallData = wall(dfloat3(0, 0, -1), (NZ - 1));
        distanceWall1 = wallData.distance +  dot_product(endpoint1, wallData.normal);
        distanceWall2 = wallData.distance +  dot_product(endpoint2, wallData.normal);

        if (distanceWall1 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall1,endpoint1,step);
        }
        if (distanceWall2 < radius) {
            capsuleWallCollisionCap(pc_i,wallData,radius-distanceWall2,endpoint2,step);
        }
    #endif
}

// ------------------------------------------------------------------------
// -------------------- COLLISION BETWEEN PARTICLES -----------------------
// ------------------------------------------------------------------------ 

__device__
void checkCollisionBetweenParticles( unsigned int column,unsigned int row,ParticleShape *shape,ParticleCenter* pc_i,ParticleCenter* pc_j,int step){

    //printf("shape %d %d \n",pc_i->collision.shape,pc_j->collision.shape);
    switch (*shape) {
        case SPHERE:
            switch (*shape) {
            case SPHERE:
            //printf("sph - sph col \n");
                //printf("collision between spheres \n");
                if(sphereSphereGap( pc_i, pc_j)<0){
                    sphereSphereCollision(column,row, pc_i, pc_j,step);
                }
                break;
            case CAPSULE:
            //printf("sphe - cap col \n");
                capsuleSphereCollisionCheck(column,row,shape,pc_i,pc_j,step);
                break;
            case ELLIPSOID:
            //printf("sph - eli col \n");
                //collision sphere-ellipsoid
                break;
            default:
            //printf("sph - def col \n");
                // Handle unknown particle types
                break;
            }
            break;
        case CAPSULE:
            switch (*shape) {
            case SPHERE:
                //printf("cap - sph col \n");
                capsuleSphereCollisionCheck(column,row,shape,pc_i,pc_j,step);
                break;
            case CAPSULE:
                //printf("cap - cap col \n");
                capsuleCapsuleCollisionCheck(column,row,pc_i,pc_j, step, pc_i->getSemiAxis1(),pc_i->getSemiAxis2(), pc_i->getRadius(),pc_j->getSemiAxis1(), pc_j->getSemiAxis2(), pc_j->getRadius());
            case ELLIPSOID:
                //printf("cap - eli col \n");
                //collision capsule-ellipsoid
                break;
            default:
                // Handle unknown particle types
                //printf("cap - def col \n");
                break;
            }
            break;
        case ELLIPSOID:
            switch (*shape) {
            case SPHERE:
                //printf("eli - sphere col \n");
                //collision ellipsoid-sphere
                break;
            case CAPSULE:
                //printf("eli - cap col \n"); 
                //collision ellipsoid-capsule
                break;
            case ELLIPSOID:
                //printf("eli - eli col \n");
                // ellipsoidEllipsoidCollisionCheck(column,row,pc_i,pc_j, step);
                break;
            default:
                //printf("eli - default col \n");
                // Handle unknown particle types
                break;
            }
            break;
        default:
            //printf("default - default col \n");
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

#endif //PARTICLE_MODEL