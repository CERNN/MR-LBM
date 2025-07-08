//#ifdef PARTICLE_MODEL

#include "particleCenter.cuh"
#include <iostream>

// Class TangentialCollisionTracker

/* Constructor */
__host__ __device__
TangentialCollisionTracker::TangentialCollisionTracker(){
    collisionIndex = -8;
    tang_length = dfloat3();
    lastCollisionStep = 0;
}

__host__ __device__  int TangentialCollisionTracker::getCollisionIndex() const { return this->collisionIndex; }
__host__ __device__  void TangentialCollisionTracker::setCollisonIndex(const unsigned int collisionIndex) { this->collisionIndex = collisionIndex; }

__host__ __device__  dfloat3 TangentialCollisionTracker::getTangLength() const { return this->tang_length; }
__host__ __device__  void TangentialCollisionTracker::setTangLength(const dfloat3& tang_length) { this->tang_length = tang_length; }

__host__ __device__  int TangentialCollisionTracker::getLastCollisionStep() const { return this->lastCollisionStep; }
__host__ __device__  void TangentialCollisionTracker::setLastCollisonStep(const unsigned int lastCollisionStep) { this->lastCollisionStep = lastCollisionStep; }



// Classs ParticleCenter 

/* Constructor */
__host__ __device__  
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

//TODO: in addition to make set and get for dfloat3, also make for each individual component

__host__ __device__  dfloat3 ParticleCenter::getPos() const { return this->pos; }
__host__ __device__  dfloat ParticleCenter::getPosX() const { return this->pos.x; }
__host__ __device__  dfloat ParticleCenter::getPosY() const { return this->pos.y; }
__host__ __device__  dfloat ParticleCenter::getPosZ() const { return this->pos.z; }
__host__ __device__  void ParticleCenter::setPos(const dfloat3 pos) { this->pos = pos; }
__host__ __device__  void ParticleCenter::setPosX(dfloat x) { this->pos.x = x; }
__host__ __device__  void ParticleCenter::setPosY(dfloat y) { this->pos.y = y; }
__host__ __device__  void ParticleCenter::setPosZ(dfloat z) { this->pos.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getPos_old() const { return this->pos_old; }
__host__ __device__ dfloat ParticleCenter::getPosOldX() const { return this->pos_old.x; }
__host__ __device__ dfloat ParticleCenter::getPosOldY() const { return this->pos_old.y; }
__host__ __device__ dfloat ParticleCenter::getPosOldZ() const { return this->pos_old.z; }
__host__ __device__ void ParticleCenter::setPos_old(const dfloat3 pos_old) { this->pos_old = pos_old; }
__host__ __device__ void ParticleCenter::setPosOldX(dfloat x) { this->pos_old.x = x; }
__host__ __device__ void ParticleCenter::setPosOldY(dfloat y) { this->pos_old.y = y; }
__host__ __device__ void ParticleCenter::setPosOldZ(dfloat z) { this->pos_old.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getVel() const { return this->vel; }
__host__ __device__ dfloat ParticleCenter::getVelX() const { return this->vel.x; }
__host__ __device__ dfloat ParticleCenter::getVelY() const { return this->vel.y; }
__host__ __device__ dfloat ParticleCenter::getVelZ() const { return this->vel.z; }
__host__ __device__ void ParticleCenter::setVel(const dfloat3& vel) { this->vel = vel; }
__host__ __device__ void ParticleCenter::setVelX(dfloat x) { this->vel.x = x; }
__host__ __device__ void ParticleCenter::setVelY(dfloat y) { this->vel.y = y; }
__host__ __device__ void ParticleCenter::setVelZ(dfloat z) { this->vel.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getVel_old() const { return this->vel_old; }
__host__ __device__ dfloat ParticleCenter::getVelOldX() const { return this->vel_old.x; }
__host__ __device__ dfloat ParticleCenter::getVelOldY() const { return this->vel_old.y; }
__host__ __device__ dfloat ParticleCenter::getVelOldZ() const { return this->vel_old.z; }
__host__ __device__ void ParticleCenter::setVel_old(const dfloat3& vel_old) { this->vel_old = vel_old; }
__host__ __device__ void ParticleCenter::setVelOldX(dfloat x) { this->vel_old.x = x; }
__host__ __device__ void ParticleCenter::setVelOldY(dfloat y) { this->vel_old.y = y; }
__host__ __device__ void ParticleCenter::setVelOldZ(dfloat z) { this->vel_old.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getW() const { return this->w; }
__host__ __device__ dfloat ParticleCenter::getWX() const { return this->w.x; }
__host__ __device__ dfloat ParticleCenter::getWY() const { return this->w.y; }
__host__ __device__ dfloat ParticleCenter::getWZ() const { return this->w.z; }
__host__ __device__ void ParticleCenter::setW(const dfloat3& w) {this->w = w; }
__host__ __device__ void ParticleCenter::setWX(dfloat x) { this->w.x = x; }
__host__ __device__ void ParticleCenter::setWY(dfloat y) { this->w.y = y; }
__host__ __device__ void ParticleCenter::setWZ(dfloat z) { this->w.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getW_avg() const { return this->w_avg; }
__host__ __device__ dfloat ParticleCenter::getWAvgX() const { return this->w_avg.x; }
__host__ __device__ dfloat ParticleCenter::getWAvgY() const { return this->w_avg.y; }
__host__ __device__ dfloat ParticleCenter::getWAvgZ() const { return this->w_avg.z; }
__host__ __device__ void ParticleCenter::setW_avg(const dfloat3& w_avg) { this->w_avg = w_avg; }
__host__ __device__ void ParticleCenter::setWAvgX(dfloat x) { this->w_avg.x = x; }
__host__ __device__ void ParticleCenter::setWAvgY(dfloat y) { this->w_avg.y = y; }
__host__ __device__ void ParticleCenter::setWAvgZ(dfloat z) { this->w_avg.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getW_old() const { return this->w_old; }
__host__ __device__ dfloat ParticleCenter::getWOldX() const { return this->w_old.x; }
__host__ __device__ dfloat ParticleCenter::getWOldY() const { return this->w_old.y; }
__host__ __device__ dfloat ParticleCenter::getWOldZ() const { return this->w_old.z; }
__host__ __device__ void ParticleCenter::setW_old(const dfloat3& w_old) { this->w_old = w_old; }
__host__ __device__ void ParticleCenter::setWOldX(dfloat x) { this->w_old.x = x; }
__host__ __device__ void ParticleCenter::setWOldY(dfloat y) { this->w_old.y = y; }
__host__ __device__ void ParticleCenter::setWOldZ(dfloat z) { this->w_old.z = z; }


__host__ __device__ dfloat3 ParticleCenter::getW_pos() const { return this->w_pos; }
__host__ __device__ dfloat ParticleCenter::getWPosX() const { return this->w_pos.x; }
__host__ __device__ dfloat ParticleCenter::getWPosY() const { return this->w_pos.y; }
__host__ __device__ dfloat ParticleCenter::getWPosZ() const { return this->w_pos.z; }
__host__ __device__ void ParticleCenter::setW_pos(const dfloat3& w_pos) { this->w_pos = w_pos; }
__host__ __device__ void ParticleCenter::setWPosX(dfloat x) { this->w_pos.x = x; }
__host__ __device__ void ParticleCenter::setWPosY(dfloat y) { this->w_pos.y = y; }
__host__ __device__ void ParticleCenter::setWPosZ(dfloat z) { this->w_pos.z = z; }


__host__ __device__ dfloat4 ParticleCenter::getQ_pos() const { return this->q_pos; }
__host__ __device__ dfloat ParticleCenter::getQPosX() const { return this->q_pos.x; }
__host__ __device__ dfloat ParticleCenter::getQPosY() const { return this->q_pos.y; }
__host__ __device__ dfloat ParticleCenter::getQPosZ() const { return this->q_pos.z; }
__host__ __device__ dfloat ParticleCenter::getQPosW() const { return this->q_pos.w; }
__host__ __device__ void ParticleCenter::setQ_pos(const dfloat4& q_pos) { this->q_pos = q_pos; }
__host__ __device__ void ParticleCenter::setQPosX(dfloat x) { this->q_pos.x = x; }
__host__ __device__ void ParticleCenter::setQPosY(dfloat y) { this->q_pos.y = y; }
__host__ __device__ void ParticleCenter::setQPosZ(dfloat z) { this->q_pos.z = z; }
__host__ __device__ void ParticleCenter::setQPosW(dfloat w) { this->q_pos.w = w; }


__host__ __device__ dfloat4 ParticleCenter::getQ_pos_old() const { return this->q_pos_old; }
__host__ __device__ dfloat ParticleCenter::getQPosOldX() const { return this->q_pos_old.x; }
__host__ __device__ dfloat ParticleCenter::getQPosOldY() const { return this->q_pos_old.y; }
__host__ __device__ dfloat ParticleCenter::getQPosOldZ() const { return this->q_pos_old.z; }
__host__ __device__ dfloat ParticleCenter::getQPosOldW() const { return this->q_pos_old.w; }
__host__ __device__ void ParticleCenter::setQ_pos_old(const dfloat4& q_pos_old) { this->q_pos_old = q_pos_old; }
__host__ __device__ void ParticleCenter::setQPosOldX(dfloat x) { this->q_pos_old.x = x; }
__host__ __device__ void ParticleCenter::setQPosOldY(dfloat y) { this->q_pos_old.y = y; }
__host__ __device__ void ParticleCenter::setQPosOldZ(dfloat z) { this->q_pos_old.z = z; }
__host__ __device__ void ParticleCenter::setQPosOldW(dfloat w) { this->q_pos_old.w = w; }

__host__ __device__ dfloat3 ParticleCenter::getF() const { return this->f; }
__host__ __device__ dfloat ParticleCenter::getFX() const { return this->f.x; }
__host__ __device__ dfloat ParticleCenter::getFY() const { return this->f.y; }
__host__ __device__ dfloat ParticleCenter::getFZ() const { return this->f.z; }
__host__ __device__ void ParticleCenter::setF(const dfloat3& f) { this->f = f; }
__host__ __device__ void ParticleCenter::setFX(dfloat x) { this->f.x = x; }
__host__ __device__ void ParticleCenter::setFY(dfloat y) { this->f.y = y; }
__host__ __device__ void ParticleCenter::setFZ(dfloat z) { this->f.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getF_old() const { return this->f_old; }
__host__ __device__ dfloat ParticleCenter::getFOldX() const { return this->f_old.x; }
__host__ __device__ dfloat ParticleCenter::getFOldY() const { return this->f_old.y; }
__host__ __device__ dfloat ParticleCenter::getFOldZ() const { return this->f_old.z; }
__host__ __device__ void ParticleCenter::setF_old(const dfloat3& f_old) { this->f_old = f_old; }
__host__ __device__ void ParticleCenter::setFOldX(dfloat x) { this->f_old.x = x; }
__host__ __device__ void ParticleCenter::setFOldY(dfloat y) { this->f_old.y = y; }
__host__ __device__ void ParticleCenter::setFOldZ(dfloat z) { this->f_old.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getM() const { return this->M; }
__host__ __device__ dfloat ParticleCenter::getMX() const { return this->M.x; }
__host__ __device__ dfloat ParticleCenter::getMY() const { return this->M.y; }
__host__ __device__ dfloat ParticleCenter::getMZ() const { return this->M.z; }
__host__ __device__ void ParticleCenter::setM(const dfloat3& M) { this->M = M; }
__host__ __device__ void ParticleCenter::setMX(dfloat x) { this->M.x = x; }
__host__ __device__ void ParticleCenter::setMY(dfloat y) { this->M.y = y; }
__host__ __device__ void ParticleCenter::setMZ(dfloat z) { this->M.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getM_old() const { return this->M_old; }
__host__ __device__ dfloat ParticleCenter::getMOldX() const { return this->M_old.x; }
__host__ __device__ dfloat ParticleCenter::getMOldY() const { return this->M_old.y; }
__host__ __device__ dfloat ParticleCenter::getMOldZ() const { return this->M_old.z; }
__host__ __device__ void ParticleCenter::setM_old(const dfloat3& M_old) { this->M_old = M_old; }
__host__ __device__ void ParticleCenter::setMOldX(dfloat x) { this->M_old.x = x; }
__host__ __device__ void ParticleCenter::setMOldY(dfloat y) { this->M_old.y = y; }
__host__ __device__ void ParticleCenter::setMOldZ(dfloat z) { this->M_old.z = z; }

__host__ __device__ dfloat6 ParticleCenter::getI() const { return this->I; }
__host__ __device__ dfloat ParticleCenter::getIXX() const { return this->I.xx; }
__host__ __device__ dfloat ParticleCenter::getIYY() const { return this->I.yy; }
__host__ __device__ dfloat ParticleCenter::getIZZ() const { return this->I.zz; }
__host__ __device__ dfloat ParticleCenter::getIXY() const { return this->I.xy; }
__host__ __device__ dfloat ParticleCenter::getIXZ() const { return this->I.xz; }
__host__ __device__ dfloat ParticleCenter::getIYZ() const { return this->I.yz; }
__host__ __device__ void ParticleCenter::setI(const dfloat6& I) { this->I = I; }
__host__ __device__ void ParticleCenter::setIXX(dfloat val) { this->I.xx = val; }
__host__ __device__ void ParticleCenter::setIYY(dfloat val) { this->I.yy = val; }
__host__ __device__ void ParticleCenter::setIZZ(dfloat val) { this->I.zz = val; }
__host__ __device__ void ParticleCenter::setIXY(dfloat val) { this->I.xy = val; }
__host__ __device__ void ParticleCenter::setIXZ(dfloat val) { this->I.xz = val; }
__host__ __device__ void ParticleCenter::setIYZ(dfloat val) { this->I.yz = val; }

__host__ __device__ dfloat3 ParticleCenter::getDP_internal() const { return this->dP_internal; }
__host__ __device__ dfloat ParticleCenter::getDPInternalX() const { return this->dP_internal.x; }
__host__ __device__ dfloat ParticleCenter::getDPInternalY() const { return this->dP_internal.y; }
__host__ __device__ dfloat ParticleCenter::getDPInternalZ() const { return this->dP_internal.z; }
__host__ __device__ void ParticleCenter::setDP_internal(const dfloat3& dP_internal) { this->dP_internal = dP_internal; }
__host__ __device__ void ParticleCenter::setDPInternalX(dfloat x) { this->dP_internal.x = x; }
__host__ __device__ void ParticleCenter::setDPInternalY(dfloat y) { this->dP_internal.y = y; }
__host__ __device__ void ParticleCenter::setDPInternalZ(dfloat z) { this->dP_internal.z = z; }

__host__ __device__ dfloat3 ParticleCenter::getDL_internal() const { return this->dL_internal; }
__host__ __device__ dfloat ParticleCenter::getDLInternalX() const { return this->dL_internal.x; }
__host__ __device__ dfloat ParticleCenter::getDLInternalY() const { return this->dL_internal.y; }
__host__ __device__ dfloat ParticleCenter::getDLInternalZ() const { return this->dL_internal.z; }
__host__ __device__ void ParticleCenter::setDL_internal(const dfloat3& dL_internal) { this->dL_internal = dL_internal; }
__host__ __device__ void ParticleCenter::setDLInternalX(dfloat x) { this->dL_internal.x = x; }
__host__ __device__ void ParticleCenter::setDLInternalY(dfloat y) { this->dL_internal.y = y; }
__host__ __device__ void ParticleCenter::setDLInternalZ(dfloat z) { this->dL_internal.z = z; }

__host__ __device__ dfloat ParticleCenter::getS() const { return this->S; }
__host__ __device__ void ParticleCenter::setS(dfloat S) { this->S = S; }

__host__ __device__ dfloat ParticleCenter::getRadius() const { return this->radius; }
__host__ __device__ void ParticleCenter::setRadius(dfloat radius) { this->radius = radius; }

__host__ __device__ dfloat ParticleCenter::getVolume() const { return this->volume; }
__host__ __device__ void ParticleCenter::setVolume(dfloat volume) { this->volume = volume; }

__host__ __device__ dfloat ParticleCenter::getDensity() const { return this->density; }
__host__ __device__ void ParticleCenter::setDensity(dfloat density) { this->density = density; }

__host__ __device__ bool ParticleCenter::getMovable() const { return this->movable; }
__host__ __device__ void ParticleCenter::setMovable(bool movable) { this->movable = movable; }

__host__ __device__ CollisionData ParticleCenter::getCollision() const { return this->collision; }
__host__ __device__ void ParticleCenter::setCollision(const CollisionData& collision) { this->collision = collision; }

//#endif //PARTICLE_MODEL