#ifndef __PARTICLE_CENTER_H
#define __PARTICLE_CENTER_H

#include "../../globalStructs.h"
#include "../../var.h"
//#include "../models/ibm/ibmVar.h"
#include "../models/dem/collisionVar.h"


typedef struct collisionData {
    dfloat3 semiAxis;
    dfloat3 semiAxis2;
    dfloat3 semiAxis3;

    // Arrays to store active collisions and their displacements
    int collisionPartnerIDs[MAX_ACTIVE_COLLISIONS];
    dfloat3 tangentialDisplacements[MAX_ACTIVE_COLLISIONS];
    int lastCollisionStep[MAX_ACTIVE_COLLISIONS];// last time step of collision
}CollisionData;

class TangentialCollisionTracker 
{    
    public:
    /* Constructor */
    __host__ __device__
    TangentialCollisionTracker();

    int getCollisionIndex() const;
    void setCollisonIndex(const unsigned int collisionIndex);

    dfloat3 getTangLength() const;
    void setTangLength(const dfloat3& tang_length);

    int getLastCollisionStep() const;
    void setLastCollisonStep(const unsigned int lastCollisionStep);

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
    ParticleCenter();

    dfloat3 getPos() const;
    void setPos(const dfloat3& pos);

    dfloat3 getPos_old() const;
    void setPos_old(const dfloat3& pos_old);

    dfloat3 getVel() const;
    void setVel(const dfloat3& vel);

    dfloat3 getVel_old() const;
    void setVel_old(const dfloat3& vel_old);

    dfloat3 getW() const;
    void setW(const dfloat3& w);

    dfloat3 getW_avg() const;
    void setW_avg(const dfloat3& w_avg);

    dfloat3 getW_old() const;
    void setW_old(const dfloat3& w_old);

    dfloat3 getW_pos() const;
    void setW_pos(const dfloat3& w_pos);

    dfloat4 getQ_pos() const;
    void setQ_pos(const dfloat4& q_pos);

    dfloat4 getQ_pos_old() const;
    void setQ_pos_old(const dfloat4& q_pos_old);

    dfloat3 getF() const;
    void setF(const dfloat3& f);

    dfloat3 getF_old() const;
    void setF_old(const dfloat3& f_old);

    dfloat3 getM() const;
    void setM(const dfloat3& M);

    dfloat3 getM_old() const;
    void setM_old(const dfloat3& M_old);

    dfloat6 getI() const;
    void setI(const dfloat6& I);

    dfloat3 getDP_internal() const;
    void setDP_internal(const dfloat3& dP_internal);

    dfloat3 getDL_internal() const;
    void setDL_internal(const dfloat3& dL_internal);

    dfloat getS() const;
    void setS(dfloat S);
    
    dfloat getRadius() const;
    void setRadius(dfloat radius);

    dfloat getVolume() const;
    void setVolume(dfloat volume);
    
    dfloat getDensity() const;
    void setDensity(dfloat density);
    
    bool getMovable() const;
    void setMovable(bool movable);

    CollisionData getCollision() const;
    void setCollision(const CollisionData& collision);

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
    bool movable;       // If the particle can move
    CollisionData collision;
}; 
#endif