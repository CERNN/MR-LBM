/**
*   @file Particle.cuh
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @author Ricardo de Souza
*   @brief Struct for particle
*   @version 0.4.0
*   @date 01/01/2025
*/
#ifndef __PARTICLE_H
#define __PARTICLE_H


#include "particleCenter.cuh"
#include "../models/ibm/ibmNodes.cuh"
#include <math.h>
#include <random>
#include "./../../var.h"
#include <map>
#include "./../../globalFunctions.h"

#ifdef PARTICLE_MODEL
/*
*   Struct for particle representation
*/
enum ParticleMethod {none, PIBM, IBM, TRACER};
struct MethodRange {
    int first = -1;
    int last = -1;
};


enum ParticleShape { SPHERE = 0 , CAPSULE = 1, ELLIPSOID = 2, GRID = 3, RANDOM = 4};

class Particle {
    public:
        __host__ __device__ Particle();
        __host__ Particle::~Particle();

        __host__ __device__ unsigned int getNumNodes() const;
        __host__ __device__ void setNumNodes(unsigned int numNodes);

        __host__ __device__ IbmNodes* getNode() const;
        __host__ __device__ void setNode(IbmNodes* node);

        __host__ __device__ ParticleMethod getMethod() const;
        __host__ __device__ void setMethod(ParticleMethod method);

        __host__ __device__ ParticleCenter* getPCenter() const;
        __host__ __device__ void setPCenter(ParticleCenter* pCenter);

        __host__ __device__ const bool& getCollideParticle() const;
        __host__ __device__ void setCollideParticle(const bool& collideParticle);

        __host__ __device__ const bool& getCollideWall() const;
        __host__ __device__ void setCollideWall(const bool& collideWall);

        __host__ __device__ ParticleShape* getShape() const;
        __host__ __device__ void setShape(ParticleShape* shape);

         /*
        *   @brief Create the particle in the shape of a sphere with given diameter and center
        *   @param part: particle object to override values
        *   @param diameter: sphere diameter in dfloat
        *   @param center : sphere center position
        *   @param coloumb: number of interations for coloumb optimization
        *   @param move: particle is movable or not
        *   @param density: particle density
        *   @param vel: particle velocity
        *   @param w: particle rotation velocity
        */
        // dfloat diameter, dfloat3 center, unsigned int coulomb, bool move,dfloat density = PARTICLE_DENSITY, dfloat3 vel = dfloat3(0, 0, 0), dfloat3 w = dfloat3(0, 0, 0)
        __host__
        void makeSpherePolar(ParticleCenter *praticleCenter);
        __host__
        void makeUniformBox(ParticleCenter *praticleCenter);
        __host__
        void makeRandomBox(ParticleCenter *praticleCenter);


        __host__
        void makeCapsule(ParticleCenter *praticleCenter);

        __host__
        void makeEllipsoid(ParticleCenter *praticleCenter);


        // __host__
        // void makeCapsule(dfloat diameter, dfloat3 point1, dfloat3 point2, bool move,
        //     dfloat density = PARTICLE_DENSITY, dfloat3 vel = dfloat3(0, 0, 0), dfloat3 w = dfloat3(0, 0, 0));



    private:
        unsigned int numNodes;
        ParticleMethod method;
        ParticleCenter *pCenter; // Particle center
        bool collideParticle; //false if particle collide with other Particles
        bool collideWall; //false if particle collide with walls
        ParticleShape *shape;
        IbmNodes *nodes;

        // std::vector<IbmNodes> nodeStorage;

};


/*
*   Particles representation as class of arrays (SoA) for better GPU performance
*/
class ParticlesSoA{
    public:
        __host__  ParticlesSoA(); // Constructor
        __host__  ~ParticlesSoA(); // Destructor

        __host__ void createParticles(Particle *particles);
        __host__ void updateParticlesAsSoA(Particle *particles);

        void updateNodesGPUs();
        void freeNodesAndCenters();

        __host__ __device__ IbmNodesSoA* getNodesSoA();
        __host__ __device__ void setNodesSoA(const IbmNodesSoA* nodes);

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

        __host__ const MethodRange& ParticlesSoA::getMethodRange(ParticleMethod method) const;
        __host__ void ParticlesSoA::setMethodRange(ParticleMethod method, int first, int last);
        __host__ int ParticlesSoA::getMethodCount(ParticleMethod method) const;

        //void updateNodesGPUs();
        //void freeNodesAndCenters();
    private:
        IbmNodesSoA nodesSoA[N_GPUS];
        ParticleCenter* pCenterArray;
        dfloat3* pCenterLastPos;    // Last particle position
        dfloat3* pCenterLastWPos;   // Last angular particle position
        ParticleShape* pShape;
        ParticleMethod* pMethod;
        bool* pCollideWall;
        bool* pCollideParticle;
        std::map<ParticleMethod, MethodRange> methodRanges;

        std::vector<ParticleCenter> centerStorage;
};

#endif //PARTICLE_MODEL
#endif

