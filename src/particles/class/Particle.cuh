#ifndef __PARTICLE_H
#define __PARTICLE_H


#include "ParticleCenter.cuh"
#include <math.h>
#include <random>
#include "./../../var.h"

/*
*   Struct for particle representation
*/
enum ParticleMethod {none, PIBM, IBM, TRACER};

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


    private:
        ParticleMethod method;
        ParticleCenter pCenter; // Particle center
        bool collideParticle; //false if particle collide with other Particles
        bool collideWall; //false if particle collide with walls

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

        //void updateNodesGPUs();
        //void freeNodesAndCenters();
    private:
        // ParticleNodeSoA* nodesSoA;
        ParticleCenter* pCenterArray;
        dfloat3* pCenterLastPos;    // Last particle position
        dfloat3* pCenterLastWPos;   // Last angular particle position
};

#endif