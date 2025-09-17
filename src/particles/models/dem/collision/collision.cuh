/**
 *  @file collision.cuh
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

//collision tracking
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

//collision mechanics with walls

/**
 *  @brief Handle collision mechanics between a sphere and a wall.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing sphere information.
 *  @param wallData: The data structure representing the wall.
 *  @param displacement: The displacement value representing how far the sphere has moved.
 *  @param step: The current time step for collision processing.
 */
__device__
void sphereWallCollision(ParticleCenter* pc_i,Wall wallData,dfloat displacement,int step);

/**
 *  @brief Handle collision mechanics between a capsule's end cap and a wall.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing capsule information.
 *  @param wallData: The data structure representing the wall.
 *  @param displacement: The displacement value for the capsule's end cap.
 *  @param endpoint: The endpoint of the capsule's end cap.
 *  @param step: The current time step for collision processing.
 */
__device__
void capsuleWallCollisionCap(ParticleCenter* pc_i,Wall wallData,dfloat displacement,dfloat3 endpoint, int step);

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
void ellipsoidWallCollision(ParticleCenter* pc_i,Wall wallData,dfloat displacement,dfloat3 endpoint, dfloat cr[1],int step);

//sphere functions
/**
 *  @brief Compute the gap between two spheres.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
 *  @return The distance between the surfaces of the two spheres.
 */
__device__
dfloat sphereSphereGap(ParticleCenter*  pc_i, ParticleCenter*  pc_j);

/**
 *  @brief Compute the shortest distance from a point to a segment considering periodic conditions.
 *  @param point: The point in 3D space.
 *  @param segStart: The start point of the segment.
 *  @param segEnd: The end point of the segment.
 *  @param closestPoint: Output for the closest point on the segment.
 *  @return The shortest distance between the point and the segment.
 */
__device__
dfloat point_to_segment_distance_periodic(dfloat3 p, dfloat3 segA, dfloat3 segB, dfloat3 closestOnAB[1]);

/**
 *  @brief Constrain a point to lie within a given segment.
 *  @param point: The point to be constrained.
 *  @param segStart: The start point of the segment.
 *  @param segEnd: The end point of the segment.
 *  @return The constrained point that lies on the segment.
 */
__device__
dfloat3 constrain_to_segment(dfloat3 point, dfloat3 segStart, dfloat3 segEnd);


/**
 *  @brief Compute the closest points and distance between two line segments in 3D.
 *  @param p1: Start point of the first segment.
 *  @param q1: End point of the first segment.
 *  @param p2: Start point of the second segment.
 *  @param q2: End point of the second segment.
 *  @param closestOnAB: Output for the closest point on the first segment.
 *  @param closestOnCD: Output for the closest point on the second segment.
 *  @return The shortest distance between the two segments.
 *  @obs: https://zalo.github.io/blog/closest-point-between-segments/
 */
__device__
dfloat segment_segment_closest_points(dfloat3 p1, dfloat3 q1, dfloat3 p2, dfloat3 q2, dfloat3 closestOnAB[1], dfloat3 closestOnCD[1]);
/**
 *  @brief Compute the closest points and distance between two line segments in 3D considering periodic conditions.
 *  @param p1: Start point of the first segment.
 *  @param q1: End point of the first segment.
 *  @param p2: Start point of the second segment.
 *  @param q2: End point of the second segment.
 *  @param closestOnAB: Output for the closest point on the first segment.
 *  @param closestOnCD: Output for the closest point on the second segment.
 *  @return The shortest distance between the two segments.
 */
__device__
dfloat segment_segment_closest_points_periodic(dfloat3 p1, dfloat3 q1, dfloat3 p2, dfloat3 q2, dfloat3 closestOnAB[1], dfloat3 closestOnCD[1]);

//ellipsoid functions
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

/**
 *  @brief Handle collision mechanics between two spheres.
 *  @param column: The column index in a grid or matrix representing the particles' positions.
 *  @param row: The row index in a grid or matrix representing the particles' positions.
 *  @param pc_i: Pointer to the `ParticleCenter` structure containing information about the first sphere.
 *  @param pc_j: Pointer to the `ParticleCenter` structure containing information about the second sphere.
 *  @param step: The current time step for collision processing.
 */
__device__
void sphereSphereCollision(unsigned int column, unsigned int row, ParticleCenter* pc_i, ParticleCenter* pc_j, int step);

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


#endif //PARTICLE_MODEL
#endif // !__IBM_COLLISION_H

