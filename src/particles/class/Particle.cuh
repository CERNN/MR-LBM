#ifndef __PARTICLE_H
#define __PARTICLE_H


#include "ParticleCenter.cuh"
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
        Particle();

        ParticleMethod getMethod() const;
        void setMethod(ParticleMethod method);

        const ParticleCenter& getPCenter() const;
        void setPCenter(const ParticleCenter& pCenter);

        const bool& getCollideParticle() const;
        void setCollideParticle(const bool& collideParticle);

        const bool& getCollideWall() const;
        void setCollideWall(const bool& collideWall);

        ParticleShape getShape() const;
        void setShape(ParticleShape shape);


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
        ParticlesSoA(); // Constructor
        ~ParticlesSoA(); // Destructor
        void createParticles(Particle particles[NUM_PARTICLES]);
        void updateParticlesAsSoA(Particle* particles);

        ParticleCenter* getPCenterArray() const;
        void setPCenterArray(ParticleCenter* pArray);

        dfloat3* getPCenterLastPos() const;
        void setPCenterLastPos(dfloat3* pLastPos);

        dfloat3* getPCenterLastWPos() const;
        void setPCenterLastWPos(dfloat3* pLastWPos);

        ParticleShape* getPShape() const;
        void setPShape(ParticleShape* pShape);

        ParticleMethod* getPMethod() const;
        void setPMethod(ParticleMethod* pMethod);

        bool* getPCollideWall() const;
        void setPCollideWall(bool* pCollideWall);

        bool* getPCollideParticle() const;
        void setPCollideParticle(bool* pCollideParticle);

        const MethodRange& ParticlesSoA::getMethodRange(ParticleMethod method) const;
        void ParticlesSoA::setMethodRange(ParticleMethod method, int first, int last);
        int ParticlesSoA::getMethodCount(ParticleMethod method) const;




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