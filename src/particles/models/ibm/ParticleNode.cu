/*
*   @file particleCenter.h
*   @author Marco Aurelio Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Ricardo de Souza (rsouza.1996@alunos.utfpr.edu.br)
*   @brief Class for IBM particle center
*   @version 0.3.0
*   @date 16/05/2025
*/

#include "ParticleNode.hpp"

// Definitions class ParticleNode

ParticleNode::ParticleNode()
    : pos(), vel(), vel_old(), f(), deltaF(), S() {}

dfloat3 ParticleNode::getPos() const { return this->pos; }
void ParticleNode::setPos(const dfloat3& Pos) { this->pos = Pos; }

dfloat3 ParticleNode::getVel() const { return this->vel; }
void ParticleNode::setVel(const dfloat3& vel) { this->vel = vel; }

dfloat3 ParticleNode::getVelOld() const { return this->vel_old; }
void ParticleNode::setVelOld(const dfloat3& vel_old) { this->vel_old = vel_old; }

dfloat3 ParticleNode::getF() const { return this->f; }
void ParticleNode::setF(const dfloat3& f) { this->f = f; }

dfloat3 ParticleNode::getDeltaF() const { return this->deltaF; }
void ParticleNode::setDeltaF(const dfloat3& deltaF) { this->deltaF = deltaF; }

float ParticleNode::getS() const { return this->S; }
void ParticleNode::setS(const dfloat& S) { this->S = S; }

// Definitions class ParticleNodeSoA

__host__ __device__
ParticleNodeSoA::ParticleNodeSoA(/* args */)
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
    this->numNodes = 0;
}

__host__ __device__
ParticleNodeSoA::~ParticleNodeSoA()
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
    this->numNodes = 0;
}

unsigned int ParticleNodeSoA::getNumNodes() const { return  this->numNodes; }
void ParticleNodeSoA::setNumNodes(const int numNodes) { this->numNodes = numNodes; }

const unsigned int* ParticleNodeSoA::getParticleCenterIdx() const { return this->particleCenterIdx; }
void ParticleNodeSoA::setParticleCenterIdx(unsigned int* particleCenterIdx) { this->particleCenterIdx = particleCenterIdx; }

dfloat3SoA ParticleNodeSoA::getPos() const { return this->pos; }
void ParticleNodeSoA::setPos(const dfloat3SoA& pos) { this->pos = pos; }

dfloat3SoA ParticleNodeSoA::getVel() const { this->vel; }
void ParticleNodeSoA::setVel(const dfloat3SoA& vel) { this->vel = vel; }

dfloat3SoA ParticleNodeSoA::getVelOld() const { this->vel_old; }
void ParticleNodeSoA::setVelOld(const dfloat3SoA& vel_old) { this->vel_old = vel_old; }

dfloat3SoA ParticleNodeSoA::getF() const { this->f; }
void ParticleNodeSoA::setF(const dfloat3SoA& f) { this->f = f; }

dfloat3SoA ParticleNodeSoA::getDeltaF() const { this->deltaF; }
void ParticleNodeSoA::setDeltaF(const dfloat3SoA& deltaF) { this->deltaF = deltaF; }

float ParticleNodeSoA::getS() const { this->S; }
void ParticleNodeSoA::setS(dfloat* S) { this->S = S; }


#ifdef IBM
void ParticleNodeSoA::allocateMemory(unsigned int numMaxNodes)
{
    this->pos.allocateMemory((size_t) numMaxNodes);
    this->vel.allocateMemory((size_t) numMaxNodes);
    this->vel_old.allocateMemory((size_t) numMaxNodes);
    this->f.allocateMemory((size_t) numMaxNodes);
    this->deltaF.allocateMemory((size_t) numMaxNodes);

    checkCudaErrors(
        cudaMallocManaged((void**)&(this->S), sizeof(dfloat) * numMaxNodes));
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->particleCenterIdx), sizeof(unsigned int) * numMaxNodes));
}

bool is_inside_gpu(dfloat3 pos, unsigned int n_gpu){
    return pos.z >= n_gpu*NZ && pos.z < (n_gpu+1)*NZ;
}

void ParticleNodeSoA::copyNodesFromParticle(Particle p, unsigned int pCenterIdx, unsigned int n_gpu)
{
    const int baseIdx = this->numNodes;
    int nodesAdded = 0;
    for (int i = 0; i < p.numNodes; i++)
    {
        if(!is_inside_gpu(p.nodes[i].pos, n_gpu))
            continue;

        this->particleCenterIdx[nodesAdded+baseIdx] = pCenterIdx;

        this->pos.copyValuesFromFloat3(p.nodes[i].pos, nodesAdded+baseIdx);
        this->vel.copyValuesFromFloat3(p.nodes[i].vel, nodesAdded+baseIdx);
        this->vel_old.copyValuesFromFloat3(p.nodes[i].vel_old, nodesAdded+baseIdx);
        this->f.copyValuesFromFloat3(p.nodes[i].f, nodesAdded+baseIdx);
        this->deltaF.copyValuesFromFloat3(p.nodes[i].deltaF, nodesAdded+baseIdx);
        this->S[nodesAdded+baseIdx] = p.nodes[i].S;
        nodesAdded += 1;
    }

    this->numNodes += nodesAdded;
}

void ParticleNodeSoA::leftShiftNodesSoA(int idx, int left_shift){
    this->particleCenterIdx[idx-left_shift] = this->particleCenterIdx[idx];
    this->S[idx-left_shift] = this->S[idx];
    this->pos.leftShift(idx, left_shift);
    this->vel.leftShift(idx, left_shift);
    this->vel_old.leftShift(idx, left_shift);
    this->f.leftShift(idx, left_shift);
    this->deltaF.leftShift(idx, left_shift);
}

#endif



