/*
*   @file particleCenter.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Ricardo de Souza (rsouza.1996@alunos.utfpr.edu.br)
*   @brief Class for IBM particle center
*   @version 0.3.0
*   @date 16/05/2025
*/

#ifndef __PARTICLE_NODE_H
#define __PARTICLE_NODE_H

#include "../../../globalStructs.h"
#include "ibmVar.h"

/*
*   Class describe the IBM node properties
*/
class ParticleNode
{

public:
    ParticleNode();

    dfloat3 getPos() const;
    void setPos(const dfloat3& pos);
   
    dfloat3 getVel() const;
    void setVel(const dfloat3& vel);
    
    dfloat3 getVelOld() const;
    void setVelOld(const dfloat3& vel_old);
   
    dfloat3 getF() const;
    void setF(const dfloat3& f);
    
    dfloat3 getDeltaF() const;
    void setDeltaF(const dfloat3& deltaF);
    
    float getS() const;
    void setS(const dfloat& S);  

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
class ParticleNodeSoA
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
    ParticleNodeSoA();
    __host__ __device__
    ~ParticleNodeSoA();

    /**
    *   @brief Allocate memory for given maximum number of nodes
    *   
    *   @param numMaxNodes: maximum number of nodes
    */
    void allocateMemory(unsigned int numMaxNodes);

    /**
    *   @brief Free allocated memory
    */
    void freeMemory();

    /**
    *   @brief Copy nodes values from particle
    *
    *   @param p: particle with nodes to copy
    *   @param pCenterIdx: index of particle center for given particle nodes
    *   @param baseIdx: base index to use while copying
    */
    void copyNodesFromParticle(struct particle p, unsigned int pCenterIdx, unsigned int n_gpu);

    void leftShiftNodesSoA(int idx, int left_shit);

    unsigned int getNumNodes() const;
    void setNumNodes(const int numNodes); 
    
    const unsigned int* getParticleCenterIdx() const;
    void setParticleCenterIdx(unsigned int* particleCenterIdx);
    
    dfloat3SoA getPos() const;
    void setPos(const dfloat3SoA& pos);
    
    dfloat3SoA getVel() const;
    void setVel(const dfloat3SoA& vel);
    
    dfloat3SoA getVelOld() const;
    void setVelOld(const dfloat3SoA& vel_old);
   
    dfloat3SoA getF() const;
    void setF(const dfloat3SoA& f);
    
    dfloat3SoA getDeltaF() const;
    void setDeltaF(const dfloat3SoA& deltaF);
   
    dfloat* getS() const;
    void setS(dfloat* S);
};


#endif