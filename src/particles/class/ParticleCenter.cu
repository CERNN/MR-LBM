#include "ParticleCenter.hpp"
#include <iostream>

// Class TangentialCollisionTracker

/* Constructor */
__host__ __device__
TangentialCollisionTracker::TangentialCollisionTracker(){
    collisionIndex = -8;
    tang_length = dfloat3();
    lastCollisionStep = 0;
}

int TangentialCollisionTracker::getCollisionIndex() const { return this->collisionIndex; }
void TangentialCollisionTracker::setCollisonIndex(const unsigned int collisionIndex) { this->collisionIndex = collisionIndex; }

dfloat3 TangentialCollisionTracker::getTangLength() const { return this->tang_length; }
void TangentialCollisionTracker::setTangLength(const dfloat3& tang_length) { this->tang_length = tang_length; }

int TangentialCollisionTracker::getLastCollisionStep() const { return this->lastCollisionStep; }
void TangentialCollisionTracker::setLastCollisonStep(const unsigned int lastCollisionStep) { this->lastCollisionStep = lastCollisionStep; }



// Classs ParticleCenter 

/* Constructor */
ParticleCenter::ParticleCenter() {
    pos = dfloat3();
    pos_old = dfloat3();
    vel = dfloat3();
    vel_old = dfloat3();
    w = dfloat3();
    w_avg = dfloat3();
    w_old = dfloat3();
    w_pos = dfloat3();
    q_pos = dfloat4();
    q_pos_old = dfloat4();
    f = dfloat3();
    f_old = dfloat3();
    M = dfloat3();
    M_old = dfloat3();
    I = dfloat6();
    dP_internal = dfloat3();
    dL_internal = dfloat3();
    S = 0;
    radius = 0;
    volume = 0;
    density = 0;
    movable = false;
}

dfloat3 ParticleCenter::getPos() const { return this->pos; }
void ParticleCenter::setPos(const dfloat3& pos) { this->pos = pos; }

dfloat3 ParticleCenter::getPos_old() const { return this->pos_old; }
void ParticleCenter::setPos_old(const dfloat3& pos_old) { this->pos_old = pos_old; }

dfloat3 ParticleCenter::getVel() const { return this->vel; }
void ParticleCenter::setVel(const dfloat3& vel) { this->vel = vel; }

dfloat3 ParticleCenter::getVel_old() const { return this->vel_old; }
void ParticleCenter::setVel_old(const dfloat3& vel_old) { this->vel_old = vel_old; }

dfloat3 ParticleCenter::getW() const { return this->w; }
void ParticleCenter::setW(const dfloat3& w) {this->w = w; }

dfloat3 ParticleCenter::getW_avg() const { return this->w_avg; }
void ParticleCenter::setW_avg(const dfloat3& w_avg) { this->w_avg = w_avg; }

dfloat3 ParticleCenter::getW_old() const { return this->w_old; }
void ParticleCenter::setW_old(const dfloat3& w_old) { this->w_old = w_old; }

dfloat3 ParticleCenter::getW_pos() const { return this->w_pos; }
void ParticleCenter::setW_pos(const dfloat3& w_pos) { this->w_pos = w_pos; }

dfloat4 ParticleCenter::getQ_pos() const { return this->q_pos; }
void ParticleCenter::setQ_pos(const dfloat4& q_pos) { this->q_pos = q_pos; }

dfloat4 ParticleCenter::getQ_pos_old() const { return this->q_pos_old; }
void ParticleCenter::setQ_pos_old(const dfloat4& q_pos_old) { this->q_pos_old = q_pos_old; }

dfloat3 ParticleCenter::getF() const { return this->f; }
void ParticleCenter::setF(const dfloat3& f) { this->f = f; }

dfloat3 ParticleCenter::getF_old() const { return this->f_old; }
void ParticleCenter::setF_old(const dfloat3& f_old) { this->f_old = f_old; }

dfloat3 ParticleCenter::getM() const { return this->M; }
void ParticleCenter::setM(const dfloat3& M) { this->M = M; }

dfloat3 ParticleCenter::getM_old() const { return this->M_old; }
void ParticleCenter::setM_old(const dfloat3& M_old) { this->M_old = M_old; }

dfloat6 ParticleCenter::getI() const { return this->I; }
void ParticleCenter::setI(const dfloat6& I) { this->I = I; }

dfloat3 ParticleCenter::getDP_internal() const { return this->dP_internal; }
void ParticleCenter::setDP_internal(const dfloat3& dP_internal) { this->dP_internal = dP_internal; }

dfloat3 ParticleCenter::getDL_internal() const { return this->dL_internal; }
void ParticleCenter::setDL_internal(const dfloat3& dL_internal) { this->dL_internal = dL_internal; }

dfloat ParticleCenter::getS() const { return this->S; }
void ParticleCenter::setS(dfloat S) { this->S = S; }

dfloat ParticleCenter::getRadius() const { return this->radius; }
void ParticleCenter::setRadius(dfloat radius) { this->radius = radius; }

dfloat ParticleCenter::getVolume() const { return this->volume; }
void ParticleCenter::setVolume(dfloat volume) { this->volume = volume; }

dfloat ParticleCenter::getDensity() const { return this->density; }
void ParticleCenter::setDensity(dfloat density) { this->density = density; }

bool ParticleCenter::getMovable() const { return this->movable; }
void ParticleCenter::setMovable(bool movable) { this->movable = movable; }

CollisionData ParticleCenter::getCollision() const { return this->collision; }
void ParticleCenter::setCollision(const CollisionData& collision) { this->collision = collision; }