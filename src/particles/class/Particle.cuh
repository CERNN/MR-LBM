//#ifdef PARTICLE_MODEL
#ifndef __PARTICLE_H
#define __PARTICLE_H


#include "particleCenter.cuh"
#include <math.h>
#include <random>
#include "./../../var.h"
#include <map>

/*
*   Struct for particle representation
*/
enum ParticleMethod {none, PIBM, IBM, TRACER};
struct MethodRange {
    int first = -1;
    int last = -1;
};


enum ParticleShape { SPHERE = 0 , CAPSULE = 1, ELLIPSOID = 2};

class Particle {
    public:
        __host__ __device__ Particle();

        __host__ __device__ ParticleMethod getMethod() const;
        __host__ __device__ void setMethod(ParticleMethod method);

        __host__ __device__ const ParticleCenter& getPCenter() const;
        __host__ __device__ void setPCenter(const ParticleCenter& pCenter);

        __host__ __device__ const bool& getCollideParticle() const;
        __host__ __device__ void setCollideParticle(const bool& collideParticle);

        __host__ __device__ const bool& getCollideWall() const;
        __host__ __device__ void setCollideWall(const bool& collideWall);

        __host__ __device__ ParticleShape getShape() const;
        __host__ __device__ void setShape(ParticleShape shape);


    private:
        ParticleMethod method;
        ParticleCenter pCenter; // Particle center
        bool collideParticle; //false if particle collide with other Particles
        bool collideWall; //false if particle collide with walls
        ParticleShape shape;

};


/*
*   Particles representation as class of arrays (SoA) for better GPU performance
*/
class ParticlesSoA{
    public:
        __host__ __device__ ParticlesSoA(); // Constructor
        __host__ __device__ ~ParticlesSoA(); // Destructor

        __host__ __device__ void createParticles(Particle *particles);
        __host__ __device__ void updateParticlesAsSoA(Particle *particles);

        __host__ __device__ ParticleCenter* getPCenterArray() const;
        __host__ __device__ void setPCenterArray(ParticleCenter* pArray);

        __host__ __device__ dfloat3* getPCenterLastPos() const;
        __host__ __device__ void setPCenterLastPos(dfloat3* pLastPos);

        __host__ __device__ dfloat3* getPCenterLastWPos() const;
        __host__ __device__ void setPCenterLastWPos(dfloat3* pLastWPos);

        __host__ __device__ ParticleShape* getPShape() const;
        __host__ __device__ void setPShape(ParticleShape* pShape);

        __host__ __device__ ParticleMethod* getPMethod() const;
        __host__ __device__ void setPMethod(ParticleMethod* pMethod);

        __host__ __device__ bool* getPCollideWall() const;
        __host__ __device__ void setPCollideWall(bool* pCollideWall);

        __host__ __device__ bool* getPCollideParticle() const;
        __host__ __device__ void setPCollideParticle(bool* pCollideParticle);

        __host__ __device__ const MethodRange& ParticlesSoA::getMethodRange(ParticleMethod method) const;
        __host__ __device__ void ParticlesSoA::setMethodRange(ParticleMethod method, int first, int last);
        __host__ __device__ int ParticlesSoA::getMethodCount(ParticleMethod method) const;




        //void updateNodesGPUs();
        //void freeNodesAndCenters();
    private:
        // ParticleNodeSoA* nodesSoA;
        ParticleCenter* pCenterArray;
        dfloat3* pCenterLastPos;    // Last particle position
        dfloat3* pCenterLastWPos;   // Last angular particle position
        ParticleShape* pShape;
        ParticleMethod* pMethod;
        bool* pCollideWall;
        bool* pCollideParticle;
        std::map<ParticleMethod, MethodRange> methodRanges;
};

#endif

//#endif