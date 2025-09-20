/**
 *  @file collision.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Handle the collision dynamics between particles
 *  @version 0.4.0
 *  @date 01/01/2025
 */

#ifndef __IBM_COLLISION_H
#define __IBM_COLLISION_H

#include "../../ibm/ibmVar.h"
#include "../../../../globalStructs.h"
#include "../../../../globalFunctions.h"
#include "../../../class/Particle.cuh"

#ifdef PARTICLE_MODEL

// ****************************************************************************
// ************************   FORCE COMPUTATION   *****************************
// ****************************************************************************

/**
 * @brief Compute the normal force during a collision.
 * @param n: The normal vector at the point of contact.
 * @param G: The relative velocity vector at the contact point.
 * @param displacement: The displacement value representing the overlap or penetration depth.
 * @param stiffness: The stiffness coefficient for the normal force calculation.
 * @param damping: The damping coefficient for the normal force calculation.
 * @return The computed normal force vector.
 */
__device__ 
dfloat3 computeNormalForce(const dfloat3& n, const dfloat3& G, dfloat displacement, dfloat stiffness, dfloat damping); 

/**
 * @brief Compute the tangential force during a collision, updating tangential displacement if slip occurs.
 * @param tang_disp: The current tangential displacement vector, which will be updated if slip occurs.
 * @param G_ct: The relative tangential velocity vector at the contact point.
 * @param stiffness: The stiffness coefficient for the tangential force calculation.
 * @param damping: The damping coefficient for the tangential force calculation.
 * @param friction_coef: The coefficient of friction between the colliding bodies.
 * @param f_n: The normal force magnitude.
 * @param t: The tangential direction vector at the contact point.
 * @param pc_i: Pointer to the ParticleCenter structure representing the particle.
 * @param tang_index: The index of the tangential displacement record for the collision.
 * @param step: The current simulation step or time index.
 * @return The computed tangential force vector.
 */
__device__ 
dfloat3 computeTangentialForce(
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
); 

/**
 * @brief Accumulate forces and torques on a particle atomically.
 * @param pc_i: Pointer to the ParticleCenter structure representing the particle.
 * @param f_dirs: The force vector to be accumulated.
 * @param m_dirs: The torque vector to be accumulated.
 */
__device__ 
void accumulateForceAndTorque(ParticleCenter* pc_i, const dfloat3& f_dirs, const dfloat3& m_dirs);

// ****************************************************************************
// ************************   COLLISION TRACKING   ****************************
// ****************************************************************************

/**
 *  @brief Calculate the index for wall collisions based on the normal vector.
 *  @param n: The normal vector of the wall.
 *  @return The calculated index used to identify wall collisions.
 */
__device__ 
int calculateWallIndex(const dfloat3 &n);
/**
 *  @brief Find the index of the collision record for a given partnerID.
 *  @param collisionData: The data structure containing collision information.
 *  @param partnerID: The ID of the collision partner to search for.
 *  @param currentTimeStep: The current time step to check collision validity.
 *  @return The index of the collision record if found, otherwise -1.
 */
__device__ 
int getCollisionIndexByPartnerID(const CollisionData &collisionData, int partnerID, int currentTimeStep);
/**
 *  @brief Start a new collision record for a given partnerID.
 *  @param collisionData: The data structure to store collision information.
 *  @param partnerID: The ID of the collision partner.
 *  @param isWall: Boolean indicating whether the collision involves a wall.
 *  @param wallNormal: The normal vector of the wall (if isWall is true).
 *  @param currentTimeStep: The current time step to record the collision.
 *  @return The index of the newly created collision record or -1 if none available.
 */
__device__ 
int startCollision(CollisionData &collisionData, int partnerID, bool isWall, const dfloat3 &wallNormal, int currentTimeStep);
/**
 *  @brief Update the tangential displacement and collision step time for a collision record.
 *  @param collisionData: The data structure containing collision information.
 *  @param index: The index of the collision record to update.
 *  @param displacement: The displacement to add to the tangential displacement.
 *  @param currentTimeStep: The current time step to update the last collision step.
 *  @return The updated tangential displacement for the collision record.
 */
__device__ 
dfloat3 updateTangentialDisplacement(CollisionData &collisionData, int index, const dfloat3 &displacement, int currentTimeStep);
/**
 *  @brief End a collision record and reset its data if necessary.
 *  @param collisionData: The data structure containing collision information.
 *  @param index: The index of the collision record to end.
 *  @param currentTimeStep: The current time step to check the validity of ending the collision.
 */
__device__ 
void endCollision(CollisionData &collisionData, int index, int currentTimeStep);


/**
 * @brief Retrieve or update the tangential displacement for a collision, starting a new collision if necessary.
 * @param pc_i: Pointer to the ParticleCenter structure representing the particle.
 * @param identifier: The ID of the collision partner (another particle or wall).
 * @param isWall: Boolean indicating whether the collision involves a wall.
 * @param step: The current simulation step or time index.
 * @param G_ct: The relative tangential velocity vector at the contact point.
 * @param G: The relative velocity vector at the contact point.
 * @param tang_index_out: Reference to an integer to store the index of the tangential displacement record.
 * @param wallNormal: The normal vector of the wall (only used if isWall is true).
 * @return The tangential displacement vector associated with the collision.
 */
__device__ 
dfloat3 getOrUpdateTangentialDisplacement(
    ParticleCenter* pc_i,
    int identifier, // wall index or partner ID
    bool isWall, //true if wall, false if particle-particle collision
    int step,
    const dfloat3& G_ct,
    const dfloat3& G,
    int& tang_index_out,
    const dfloat3& wallNormal = dfloat3{0,0,0} // only used for wall
);

// ****************************************************************************
// ****************************   WALL COLLISION   ****************************
// ****************************************************************************



/**
 *  @brief Handle collision mechanics between a sphere and a wall.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing sphere information.
 *  @param wallData: The data structure representing the wall.
 *  @param displacement: The displacement value representing how far the sphere has moved.
 *  @param step: The current time step for collision processing.
 */
__device__
void sphereWallCollision(const CollisionContext& ctx);

/**
 *  @brief Handle collision mechanics between a capsule's end cap and a wall.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing capsule information.
 *  @param wallData: The data structure representing the wall.
 *  @param displacement: The displacement value for the capsule's end cap.
 *  @param step: The current time step for collision processing.
 *  @param endpoint: The endpoint of the capsule's end cap.
 */
__device__
void capsuleWallCollisionCap(const CollisionContext& ctx);

/**
 *  @brief Handle collision mechanics between an ellipsoid particle and a wall.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param wallData: Structure containing information about the wall, such as normal and distance.
 *  @param displacement: The displacement value 
 *  @param endpoint: The point on the ellipsoid's surface where the collision occurs.
 *  @param cr: gaussian radius on the contact point
 *  @param step: The current simulation step or time index.
 */
__device__
void ellipsoidWallCollision(const CollisionContext& ctx, dfloat cr[1]);

/**
 *  @brief Calculate the distance between an ellipsoid particle and a wall, and find the contact point.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param wallData: Structure containing information about the wall, such as normal and distance.
 *  @param contactPoint2: Pointer to store the contact point between the ellipsoid and the wall.
 *  @param step: The current simulation step or time index.
 *  @param radius: gaussian radius on the closest point
 *  @return The distance between the ellipsoid and the wall at the point of contact.
 */
__device__
dfloat ellipsoidWallCollisionDistance( ParticleCenter* pc_i, Wall wallData, dfloat3* contactPoint2, dfloat radius[1], unsigned int step);

// ****************************************************************************
// ************************   PARTICLE COLLISION   ****************************
// ****************************************************************************

/**
 *  @brief Handle collision mechanics between two spheres.
 *  @param column: The column index in a grid or matrix representing the particles' positions.
 *  @param row: The row index in a grid or matrix representing the particles' positions.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
 *  @param step: The current time step for collision processing.
 */
__device__
void sphereSphereCollision(const CollisionContext& ctx);

/**
 *  @brief Handle collision mechanics between two capsules.
 *  @param column: The column index in a grid or matrix representing the particles' positions.
 *  @param row: The row index in a grid or matrix representing the particles' positions.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first capsule.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second capsule.
 *  @param closestOnA: Closest point in the axis of particle i.
 *  @param closestOnB: Closest point in the axis of particle j.
 *  @param step: The current time step for collision processing.
 */
__device__
void capsuleCapsuleCollision(unsigned int column, unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, dfloat3* closestOnA, dfloat3* closestOnB, int step);

/**
 *  @brief Process the collision between two ellipsoids by determining their closest points and applying collision response.
 *  @param column: The column index representing the position of the first ellipsoid in the grid or matrix.
 *  @param row: The row index representing the position of the second ellipsoid in the grid or matrix.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first ellipsoid.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second ellipsoid.
 *  @param closestOnA: Array to store the closest point on the surface of the first ellipsoid (A).
 *  @param closestOnB: Array to store the closest point on the surface of the second ellipsoid (B).
 *  @param dist: The calculated distance between the two ellipsoids at their closest points.
 *  @param cr1: gaussian radius on the contact point for ellipsoid 1
 *  @param cr2: gaussian radius on the contact point for ellipsoid 2
 *  @param step: The current simulation time step for collision processing.
 */
__device__
void ellipsoidEllipsoidCollision(unsigned int column, unsigned int row, ParticleCenter*  pc_i, ParticleCenter*  pc_j,dfloat3 closestOnA[1], dfloat3 closestOnB[1], dfloat dist,  dfloat cr1[1], dfloat cr2[1], dfloat3 translation, int step);

// ****************************************************************************
// ******************   AUXILIARY COLLISION FUNCTIONS  ************************
// ****************************************************************************

/**
 *  @brief Compute the intersection point between a line and the ellipsoid.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param R: 3x3 rotation matrix used to transform the ellipsoid's orientation.
 *  @param line_origin: The origin of the line used for intersection calculation.
 *  @param line_dir: The direction of the line used for intersection calculation.
 *  @return The intersection parameter between the line and the ellipsoid on .x and .y; ;.z is trash
 */
__device__
dfloat3 ellipsoid_intersection(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 line_origin, dfloat3 line_dir,dfloat3 translation);

/**
 *  @brief Compute the normal vector at a given point on the ellipsoid's surface.
 *  @param pc_i: Pointer to the ParticleCenter structure representing the ellipsoid particle.
 *  @param R: 3x3 rotation matrix used to transform the ellipsoid's orientation.
 *  @param point: The point on the ellipsoid's surface where the normal vector is computed.
 *  @param radius: gaussian radius on the point
 *  @return The normal vector at the specified point on the ellipsoid's surface.
 */
__device__
dfloat3 ellipsoid_normal(ParticleCenter* pc_i, dfloat R[3][3],dfloat3 point, dfloat radius[1],dfloat3 translation);

/**
 *  @brief Calculate the distance between two colliding ellipsoids and determine the contact points on their surfaces.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first ellipsoid.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second ellipsoid.
 *  @param contactPoint1: Array to store the computed contact point on the surface of the first ellipsoid.
 *  @param contactPoint2: Array to store the computed contact point on the surface of the second ellipsoid.
 *  @param cr1: gaussian radius on the contact point for ellipsoid 1
 *  @param cr2: gaussian radius on the contact point for ellipsoid 2
 *  @param step: The current time step for collision detection.
 *  @return The computed distance between the two ellipsoids at the contact points.
 */
__device__
dfloat ellipsoidEllipsoidCollisionDistance( ParticleCenter* pc_i, ParticleCenter* pc_j, dfloat3 contactPoint1[1], dfloat3 contactPoint2[1], dfloat cr1[1], dfloat cr2[1], dfloat3 translation, unsigned int step);

/**
 *  @brief Compute the contact points between two particles based on the given direction vector and tangent vectors.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the particle in order to determine the vector origon.
 *  @param dir: The direction vector representing the line connecting the two particles.
 *  @param t1: The contact points t value of the segment for the first ellipsoid
 *  @param t2: The contact points t value of the segment for the second ellipsoid
 *  @param contactPoint1: Array to store the first computed contact point on the surface of the particle.
 *  @param contactPoint2: Array to store the second computed contact point on the surface of the particle.
 */
__device__ 
void computeContactPoints(dfloat3 pc_i, dfloat3 dir, dfloat3 t1, dfloat3 t2, dfloat3 contactPoint1[1], dfloat3 contactPoint2[1]);




#endif //PARTICLE_MODEL
#endif // !__IBM_COLLISION_H

