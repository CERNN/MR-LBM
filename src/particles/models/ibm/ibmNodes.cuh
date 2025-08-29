//#ifdef PARTICLE_MODEL

#ifndef __IBM_NODES_H
#define __IBM_NODES_H

#include "../../../globalStructs.h"
#include "../../models/ibm/ibmVar.h"
// #include "../../class/Particle.cuh"
//#include "../../class/ParticleCenter.cuh"
#pragma once

class Particle;

/*
*   Class describe the IBM node properties
*/
class IbmNodes
{

public:
    __host__ __device__ IbmNodes();

    __host__ __device__ dfloat3 getPos() const;
    __host__ __device__ void setPos(const dfloat3& pos);
   
    __host__ __device__ dfloat3 getVel() const;
    __host__ __device__ void setVel(const dfloat3& vel);
    
    __host__ __device__ dfloat3 getVelOld() const;
    __host__ __device__ void setVelOld(const dfloat3& vel_old);
   
    __host__ __device__ dfloat3 getF() const;
    __host__ __device__ void setF(const dfloat3& f);
    
    __host__ __device__ dfloat3 getDeltaF() const;
    __host__ __device__ void setDeltaF(const dfloat3& deltaF);
    
    __host__ __device__ float getS() const;
    __host__ __device__ void setS(const dfloat& S);  

protected:
    dfloat3 pos; // node coordinate
    dfloat3 vel; // node velocity
    dfloat3 vel_old; // node last step velocity
    dfloat3 f;  // node force
    dfloat3 deltaF;  // node force variation
    dfloat S; // node surface area
};

/*
*   Class to represent the particle nodes as a Structure of Arrays, 
*   instead of a Array of Structures
*/
class IbmNodesSoA
{
protected:
    unsigned int numNodes; // number of nodes
    unsigned int* particleCenterIdx; // index of particle center for each node
    dfloat3SoA pos; // vectors with nodes coordinates
    dfloat3SoA vel; // vectors with nodes velocities
    dfloat3SoA vel_old; // vectors with nodes old velocities
    dfloat3SoA f;  // vectors with nodes forces
    dfloat3SoA deltaF;  // vectors with nodes forces variations
    dfloat* S; // vector node surface area

public:
    __host__ __device__
    IbmNodesSoA();
    __host__ __device__
    ~IbmNodesSoA();

    /**
    *   @brief Allocate memory for given maximum number of nodes
    *   
    *   @param numMaxNodes: maximum number of nodes
    */
   __host__ void allocateMemory(unsigned int numMaxNodes);

    /**
    *   @brief Free allocated memory
    */
   __host__ void freeMemory();

    /**
    *   @brief Copy nodes values from particle
    *
    *   @param p: particle with nodes to copy
    *   @param pCenterIdx: index of particle center for given particle nodes
    *   @param baseIdx: base index to use while copying
    */
    __host__ void copyNodesFromParticle(Particle *particle, unsigned int pCenterIdx, unsigned int n_gpu);
 
    __host__ __device__ void updateNodesGPUs();
    __host__ __device__ void freeNodesAndCenters();

    __host__ __device__ void leftShiftNodesSoA(int idx, int left_shit);

    __host__ __device__  unsigned int getNumNodes() const;
    __host__ __device__ void setNumNodes(const int numNodes); 
    
    __host__ __device__ const unsigned int* getParticleCenterIdx() const;
    __host__ __device__ unsigned int* getParticleCenterIdx();
    __host__ __device__ void setParticleCenterIdx(unsigned int* particleCenterIdx);
    
    __host__ __device__ dfloat3SoA getPos() const;
    __host__ __device__ void setPos(const dfloat3SoA& pos);
    
    __host__ __device__ dfloat3SoA getVel() const;
    __host__ __device__ void setVel(const dfloat3SoA& vel);
    
    __host__ __device__ dfloat3SoA getVelOld() const;
    __host__ __device__ void setVelOld(const dfloat3SoA& vel_old);
   
    __host__ __device__ dfloat3SoA getF() const;
    __host__ __device__ void setF(const dfloat3SoA& f);
    
    __host__ __device__ dfloat3SoA getDeltaF() const;
    __host__ __device__ void setDeltaF(const dfloat3SoA& deltaF);
   
    __host__ __device__ dfloat* getS() const;
    __host__ __device__ void setS(dfloat* S);
};


#endif

//#endif