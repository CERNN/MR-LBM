/**
 *  @file ParticleCenter.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief Struct for particle center
 *  @version 0.4.0
 *  @date 01/01/2025
 */

#ifndef __PARTICLE_CENTER_H
#define __PARTICLE_CENTER_H

#include "../../globalStructs.h"
#include "../../var.h"
//#include "../models/ibm/ibmVar.h"
#include "../models/dem/collision/collisionVar.h"

#ifdef PARTICLE_MODEL
class CollisionData {
    public:
        __host__ __device__ CollisionData();  // Construtor

        // Reset
        __host__ __device__ void reset();

        __host__ __device__ int getCollisionPartnerID(int idx) const;
        __host__ __device__ void setCollisionPartnerID(int idx, int id);

        __host__ __device__ dfloat3 getTangentialDisplacement(int idx) const;
        __host__ __device__ void setTangentialDisplacement(int idx, dfloat3 disp);

        __host__ __device__ int getLastCollisionStep(int idx) const;
        __host__ __device__ void setLastCollisionStep(int idx, int step);

    protected:
        int collisionPartnerIDs[MAX_ACTIVE_COLLISIONS];
        dfloat3 tangentialDisplacements[MAX_ACTIVE_COLLISIONS];
        int lastCollisionStep[MAX_ACTIVE_COLLISIONS];
};

class TangentialCollisionTracker 
{    
    public:
    /* Constructor */
    __host__ __device__
    TangentialCollisionTracker();

    __host__ __device__  int getCollisionIndex() const;
    __host__ __device__  void setCollisonIndex(const unsigned int collisionIndex);

    __host__ __device__  dfloat3 getTangLength() const;
    __host__ __device__  void setTangLength(const dfloat3& tang_length);

    __host__ __device__  int getLastCollisionStep() const;
    __host__ __device__  void setLastCollisonStep(const unsigned int lastCollisionStep);

    protected:
            /*
        -4 : Special, i.e round boundary;
        -7 : Front
        -1 : Back
        -6 : North
        -2 : South
        -3 : West
        -5 : East
        0 >= : particle ID
        */
        int collisionIndex;
        dfloat3 tang_length;
        unsigned int lastCollisionStep;
};


/*
*   Class for the particle center properties
*/
class ParticleCenter
{
public:
    __host__ __device__
    ParticleCenter();

    // Position
    __host__ __device__ dfloat3 getPos() const;
    __host__ __device__ dfloat getPosX() const;
    __host__ __device__ dfloat getPosY() const;
    __host__ __device__ dfloat getPosZ() const;
    __host__ __device__ void setPos(const dfloat3 pos);
    __host__ __device__ void setPosX(dfloat x);
    __host__ __device__ void setPosY(dfloat y);
    __host__ __device__ void setPosZ(dfloat z);

    // Old Position
    __host__ __device__ dfloat3 getPos_old() const;
    __host__ __device__ dfloat getPosOldX() const;
    __host__ __device__ dfloat getPosOldY() const;
    __host__ __device__ dfloat getPosOldZ() const;
    __host__ __device__ void setPos_old(const dfloat3 pos_old);
    __host__ __device__ void setPosOldX(dfloat x);
    __host__ __device__ void setPosOldY(dfloat y);
    __host__ __device__ void setPosOldZ(dfloat z);

    // Velocity
    __host__ __device__ dfloat3 getVel() const;
    __host__ __device__ dfloat getVelX() const;
    __host__ __device__ dfloat getVelY() const;
    __host__ __device__ dfloat getVelZ() const;
    __host__ __device__ void setVel(const dfloat3& vel);
    __host__ __device__ void setVelX(dfloat x);
    __host__ __device__ void setVelY(dfloat y);
    __host__ __device__ void setVelZ(dfloat z);

    // Old Velocity
    __host__ __device__ dfloat3 getVel_old() const;
    __host__ __device__ dfloat getVelOldX() const;
    __host__ __device__ dfloat getVelOldY() const;
    __host__ __device__ dfloat getVelOldZ() const;
    __host__ __device__ void setVel_old(const dfloat3& vel_old);
    __host__ __device__ void setVelOldX(dfloat x);
    __host__ __device__ void setVelOldY(dfloat y);
    __host__ __device__ void setVelOldZ(dfloat z);

    // Angular velocity
    __host__ __device__ dfloat3 getW() const;
    __host__ __device__ dfloat getWX() const;
    __host__ __device__ dfloat getWY() const;
    __host__ __device__ dfloat getWZ() const;
    __host__ __device__ void setW(const dfloat3& w);
    __host__ __device__ void setWX(dfloat x);
    __host__ __device__ void setWY(dfloat y);
    __host__ __device__ void setWZ(dfloat z);

    // Average angular velocity
    __host__ __device__ dfloat3 getW_avg() const;
    __host__ __device__ dfloat getWAvgX() const;
    __host__ __device__ dfloat getWAvgY() const;
    __host__ __device__ dfloat getWAvgZ() const;
    __host__ __device__ void setW_avg(const dfloat3& w_avg);
    __host__ __device__ void setWAvgX(dfloat x);
    __host__ __device__ void setWAvgY(dfloat y);
    __host__ __device__ void setWAvgZ(dfloat z);

    // Old angular velocity
    __host__ __device__ dfloat3 getW_old() const;
    __host__ __device__ dfloat getWOldX() const;
    __host__ __device__ dfloat getWOldY() const;
    __host__ __device__ dfloat getWOldZ() const;
    __host__ __device__ void setW_old(const dfloat3& w_old);
    __host__ __device__ void setWOldX(dfloat x);
    __host__ __device__ void setWOldY(dfloat y);
    __host__ __device__ void setWOldZ(dfloat z);

    // Angular position vector
    __host__ __device__ dfloat3 getW_pos() const;
    __host__ __device__ dfloat getWPosX() const;
    __host__ __device__ dfloat getWPosY() const;
    __host__ __device__ dfloat getWPosZ() const;
    __host__ __device__ void setW_pos(const dfloat3& w_pos);
    __host__ __device__ void setWPosX(dfloat x);
    __host__ __device__ void setWPosY(dfloat y);
    __host__ __device__ void setWPosZ(dfloat z);

    // Quaternion orientation
    __host__ __device__ dfloat4 getQ_pos() const;
    __host__ __device__ dfloat getQPosX() const;
    __host__ __device__ dfloat getQPosY() const;
    __host__ __device__ dfloat getQPosZ() const;
    __host__ __device__ dfloat getQPosW() const;
    __host__ __device__ void setQ_pos(const dfloat4& q_pos);
    __host__ __device__ void setQPosX(dfloat x);
    __host__ __device__ void setQPosY(dfloat y);
    __host__ __device__ void setQPosZ(dfloat z);
    __host__ __device__ void setQPosW(dfloat w);

    // Old quaternion orientation
    __host__ __device__ dfloat4 getQ_pos_old() const;
    __host__ __device__ dfloat getQPosOldX() const;
    __host__ __device__ dfloat getQPosOldY() const;
    __host__ __device__ dfloat getQPosOldZ() const;
    __host__ __device__ dfloat getQPosOldW() const;
    __host__ __device__ void setQ_pos_old(const dfloat4& q_pos_old);
    __host__ __device__ void setQPosOldX(dfloat x);
    __host__ __device__ void setQPosOldY(dfloat y);
    __host__ __device__ void setQPosOldZ(dfloat z);
    __host__ __device__ void setQPosOldW(dfloat w);

    // Force
    __host__ __device__ dfloat3 getF() const;
    __host__ __device__ dfloat getFX() const;
    __host__ __device__ dfloat getFY() const;
    __host__ __device__ dfloat getFZ() const;
    __host__ __device__ void setF(const dfloat3& f);
    __host__ __device__ void setFX(dfloat x);
    __host__ __device__ void setFY(dfloat y);
    __host__ __device__ void setFZ(dfloat z);

    __host__ __device__ dfloat& getFXatomic();
    __host__ __device__ dfloat& getFYatomic();
    __host__ __device__ dfloat& getFZatomic();

    // Old force
    __host__ __device__ dfloat3 getF_old() const;
    __host__ __device__ dfloat getFOldX() const;
    __host__ __device__ dfloat getFOldY() const;
    __host__ __device__ dfloat getFOldZ() const;
    __host__ __device__ void setF_old(const dfloat3& f_old);
    __host__ __device__ void setFOldX(dfloat x);
    __host__ __device__ void setFOldY(dfloat y);
    __host__ __device__ void setFOldZ(dfloat z);

    // Moment
    __host__ __device__ dfloat3 getM() const;
    __host__ __device__ dfloat getMX() const;
    __host__ __device__ dfloat getMY() const;
    __host__ __device__ dfloat getMZ() const;
    __host__ __device__ void setM(const dfloat3& M);
    __host__ __device__ void setMX(dfloat x);
    __host__ __device__ void setMY(dfloat y);
    __host__ __device__ void setMZ(dfloat z);

    __host__ __device__ dfloat& getMXatomic();
    __host__ __device__ dfloat& getMYatomic();
    __host__ __device__ dfloat& getMZatomic();

    // Old moment
    __host__ __device__ dfloat3 getM_old() const;
    __host__ __device__ dfloat getMOldX() const;
    __host__ __device__ dfloat getMOldY() const;
    __host__ __device__ dfloat getMOldZ() const;
    __host__ __device__ void setM_old(const dfloat3& M_old);
    __host__ __device__ void setMOldX(dfloat x);
    __host__ __device__ void setMOldY(dfloat y);
    __host__ __device__ void setMOldZ(dfloat z);

    // Inertia tensor
    __host__ __device__ dfloat6 getI() const;
    __host__ __device__ dfloat getIXX() const;
    __host__ __device__ dfloat getIYY() const;
    __host__ __device__ dfloat getIZZ() const;
    __host__ __device__ dfloat getIXY() const;
    __host__ __device__ dfloat getIXZ() const;
    __host__ __device__ dfloat getIYZ() const;
    __host__ __device__ void setI(const dfloat6& I);
    __host__ __device__ void setIXX(dfloat val);
    __host__ __device__ void setIYY(dfloat val);
    __host__ __device__ void setIZZ(dfloat val);
    __host__ __device__ void setIXY(dfloat val);
    __host__ __device__ void setIXZ(dfloat val);
    __host__ __device__ void setIYZ(dfloat val);

    // Internal momentum change
    __host__ __device__ dfloat3 getDP_internal() const;
    __host__ __device__ dfloat getDPInternalX() const;
    __host__ __device__ dfloat getDPInternalY() const;
    __host__ __device__ dfloat getDPInternalZ() const;
    __host__ __device__ void setDP_internal(const dfloat3& dP_internal);
    __host__ __device__ void setDPInternalX(dfloat x);
    __host__ __device__ void setDPInternalY(dfloat y);
    __host__ __device__ void setDPInternalZ(dfloat z);

    // Internal angular momentum change
    __host__ __device__ dfloat3 getDL_internal() const;
    __host__ __device__ dfloat getDLInternalX() const;
    __host__ __device__ dfloat getDLInternalY() const;
    __host__ __device__ dfloat getDLInternalZ() const;
    __host__ __device__ void setDL_internal(const dfloat3& dL_internal);
    __host__ __device__ void setDLInternalX(dfloat x);
    __host__ __device__ void setDLInternalY(dfloat y);
    __host__ __device__ void setDLInternalZ(dfloat z);


    __host__ __device__ dfloat getS() const;
    __host__ __device__ void setS(dfloat S);
    
    __host__ __device__ dfloat getRadius() const;
    __host__ __device__ void setRadius(dfloat radius);

    __host__ __device__ dfloat getVolume() const;
    __host__ __device__ void setVolume(dfloat volume);
    
    __host__ __device__ dfloat getDensity() const;
    __host__ __device__ void setDensity(dfloat density);

    __host__ __device__ dfloat getDiameter() const;
    __host__ __device__ void setDiameter(dfloat diameter);
    
    __host__ __device__ bool getMovable() const;
    __host__ __device__ void setMovable(bool movable);

    __host__ __device__ CollisionData& getCollision()   ;
    __host__ __device__ void setCollision(const CollisionData& collision);

    __host__ __device__ dfloat3 getSemiAxis1() const;
    __host__ __device__ dfloat getSemiAxis1X() const;
    __host__ __device__ dfloat getSemiAxis1Y() const;
    __host__ __device__ dfloat getSemiAxis1Z() const;
    __host__ __device__ void setSemiAxis1(const dfloat3& semiAxis1);
    __host__ __device__ void setSemiAxis1X(dfloat x);
    __host__ __device__ void setSemiAxis1Y(dfloat y);
    __host__ __device__ void setSemiAxis1Z(dfloat z);

    __host__ __device__ dfloat3 getSemiAxis2() const;
    __host__ __device__ dfloat getSemiAxis2X() const;
    __host__ __device__ dfloat getSemiAxis2Y() const;
    __host__ __device__ dfloat getSemiAxis2Z() const;
    __host__ __device__ void setSemiAxis2(const dfloat3& semiAxis2);
    __host__ __device__ void setSemiAxis2X(dfloat x);
    __host__ __device__ void setSemiAxis2Y(dfloat y);
    __host__ __device__ void setSemiAxis2Z(dfloat z);

    __host__ __device__ dfloat3 getSemiAxis3() const;
    __host__ __device__ dfloat getSemiAxis3X() const;
    __host__ __device__ dfloat getSemiAxis3Y() const;
    __host__ __device__ dfloat getSemiAxis3Z() const;
    __host__ __device__ void setSemiAxis3(const dfloat3& semiAxis3);
    __host__ __device__ void setSemiAxis3X(dfloat x);
    __host__ __device__ void setSemiAxis3Y(dfloat y);
    __host__ __device__ void setSemiAxis3Z(dfloat z);

    // __host__ __device__ dfloat3 getPos() const;
    // __host__ __device__ dfloat getPosX() const;
    // __host__ __device__ dfloat getPosY() const;
    // __host__ __device__ dfloat getPosZ() const;
    // __host__ __device__ void setPos(const dfloat3 pos);
    __host__ __device__ dfloat3 getCenter1() const;
    __host__ __device__ void setCenter1(const dfloat3 center1);
    __host__ __device__ dfloat3 getCenter2() const;
    __host__ __device__ void setCenter2(const dfloat3 center2);

protected:
    dfloat3 pos;        // Particle center position
    dfloat3 pos_old;    // Old Particle center position
    dfloat3 vel;        // Particle center translation velocity
    dfloat3 vel_old;    // Old particle center translation velocity
    dfloat3 w;          // Particle center rotation velocity
    dfloat3 w_avg;      // Average particle rotation (used by nodes in movement)
    dfloat3 w_old;      // Old particle center rotation velocity
    dfloat3 w_pos;      // Particle angular position
    dfloat4 q_pos; // Particle angular poistion defined by a quartenion
    dfloat4 q_pos_old; // Particle angular poistion defined by a quartenion
    dfloat3 f;          // Sum of the forces acting on particle
    dfloat3 f_old;      // Old sum of the forces acting on particle
    dfloat3 M;          // Total momentum acting on particle
    dfloat3 M_old;      // Old total momentum acting on particle
    dfloat6 I;          // I innertia moment I.x = Ixx
    dfloat3 dP_internal; // Linear momentum of fluid mass inside IBM particle mesh (delta - backward Euler)
    dfloat3 dL_internal; // Angular momentum of fluid mass inside IBM particle mesh (delta - backward Euler)
    dfloat S;           // Total area of the particle
    dfloat radius;      // Sphere radius
    dfloat volume;      // Particle volume
    dfloat density;     // Particle density
    dfloat diameter;    // Particle diameter
    dfloat3 center1;
    dfloat3 center2;
    bool movable;       // If the particle can move
    dfloat3 semiAxis1;
    dfloat3 semiAxis2;
    dfloat3 semiAxis3;
    CollisionData collision;
}; 


struct CollisionContext {
    ParticleCenter* pc_i;      // Main particle (i)
    Wall wall;                 // Wall info (for wall collisions)
    dfloat displacement;       // Overlap/displacement
    unsigned int step;         // Current timestep
    ParticleCenter* pc_j;      // Partner particle (j), nullptr for wall collisions
    dfloat3 contactPoint;      // For capsule/ellipsoid
    int partnerID;             // Partner particle ID (for pairwise collisions)
};



#endif //PARTICLE_MODEL
#endif //!__PARTICLE_CENTER_H

