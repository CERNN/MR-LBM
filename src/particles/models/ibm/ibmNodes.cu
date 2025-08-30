
//#ifdef PARTICLE_MODEL

#include "IbmNodes.cuh"
#include "../../class/Particle.cuh"
// #include "../../class/ParticleCenter.cuh"

// Definitions class IbmNodes
__host__ __device__
IbmNodes::IbmNodes()
    : pos(), vel(), vel_old(), f(), deltaF(), S() {}

__host__ __device__ dfloat3 IbmNodes::getPos() const { return this->pos; }
__host__ __device__ void IbmNodes::setPos(const dfloat3& Pos) { this->pos = Pos; }

__host__ __device__ dfloat3 IbmNodes::getVel() const { return this->vel; }
__host__ __device__ void IbmNodes::setVel(const dfloat3& vel) { this->vel = vel; }

__host__ __device__ dfloat3 IbmNodes::getVelOld() const { return this->vel_old; }
__host__ __device__ void IbmNodes::setVelOld(const dfloat3& vel_old) { this->vel_old = vel_old; }

__host__ __device__ dfloat3 IbmNodes::getF() const { return this->f; }
__host__ __device__ void IbmNodes::setF(const dfloat3& f) { this->f = f; }

__host__ __device__ dfloat3 IbmNodes::getDeltaF() const { return this->deltaF; }
__host__ __device__ void IbmNodes::setDeltaF(const dfloat3& deltaF) { this->deltaF = deltaF; }

__host__ __device__ float IbmNodes::getS() const { return this->S; }
__host__ __device__ void IbmNodes::setS(const dfloat& S) { this->S = S; }

// Definitions class IbmNodesSoA

__host__ __device__
IbmNodesSoA::IbmNodesSoA(/* args */)
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
    this->numNodes = 0;
}

__host__ __device__
IbmNodesSoA::~IbmNodesSoA()
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
    this->numNodes = 0;
}

__host__ __device__ unsigned int IbmNodesSoA::getNumNodes() const { return  this->numNodes; }
__host__ __device__ void IbmNodesSoA::setNumNodes(const int numNodes) { this->numNodes = numNodes; }

__host__ __device__ const unsigned int* IbmNodesSoA::getParticleCenterIdx() const { return this->particleCenterIdx; }
__host__ __device__ unsigned int* IbmNodesSoA::getParticleCenterIdx() {return this->particleCenterIdx;}
__host__ __device__ void IbmNodesSoA::setParticleCenterIdx(unsigned int* particleCenterIdx) { this->particleCenterIdx = particleCenterIdx; }

__host__ __device__ dfloat3SoA IbmNodesSoA::getPos() const { return this->pos; }
__host__ __device__ void IbmNodesSoA::setPos(const dfloat3SoA& pos) { this->pos = pos; }

__host__ __device__ dfloat3SoA IbmNodesSoA::getVel() const { return this->vel; }
__host__ __device__ void IbmNodesSoA::setVel(const dfloat3SoA& vel) { this->vel = vel; }

__host__ __device__ dfloat3SoA IbmNodesSoA::getVelOld() const { return this->vel_old; }
__host__ __device__ void IbmNodesSoA::setVelOld(const dfloat3SoA& vel_old) { this->vel_old = vel_old; }

__host__ __device__ dfloat3SoA IbmNodesSoA::getF() const { return this->f; }
__host__ __device__ void IbmNodesSoA::setF(const dfloat3SoA& f) { this->f = f; }

__host__ __device__ dfloat3SoA IbmNodesSoA::getDeltaF() const { return this->deltaF; }
__host__ __device__ void IbmNodesSoA::setDeltaF(const dfloat3SoA& deltaF) { this->deltaF = deltaF; }

__host__ __device__ dfloat* IbmNodesSoA::getS() const { return this->S; }
__host__ __device__ void IbmNodesSoA::setS(dfloat* S) { this->S = S; }


__host__
void IbmNodesSoA::allocateMemory(unsigned int numMaxNodes)
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

__host__
void IbmNodesSoA::freeMemory()
{
    this->numNodes = 0;

    this->pos.freeMemory();
    this->vel.freeMemory();
    this->vel_old.freeMemory();
    this->f.freeMemory();
    this->deltaF.freeMemory();

    cudaFree(this->S);
    cudaFree(this->particleCenterIdx);
}

__host__ __device__
bool is_inside_gpu(dfloat3 pos, unsigned int n_gpu){
    return pos.z >= n_gpu*NZ && pos.z < (n_gpu+1)*NZ;
}

__host__
void IbmNodesSoA::copyNodesFromParticle(Particle *p, unsigned int pCenterIdx, unsigned int n_gpu)
{
    const int baseIdx = this->numNodes;
    int nodesAdded = 0;
    IbmNodes* node = p->getNode();

    for (int i = 0; i < p->getNumNodes(); i++)
    {

        if(!is_inside_gpu(node[i].getPos(), n_gpu))
            continue;

        this->particleCenterIdx[nodesAdded+baseIdx] = pCenterIdx;

        this->pos.copyValuesFromFloat3(node[i].getPos(), nodesAdded+baseIdx);
        this->vel.copyValuesFromFloat3(node[i].getVel(), nodesAdded+baseIdx);
        this->vel_old.copyValuesFromFloat3(node[i].getVelOld(), nodesAdded+baseIdx);
        this->f.copyValuesFromFloat3(node[i].getF(), nodesAdded+baseIdx);
        this->deltaF.copyValuesFromFloat3(node[i].getDeltaF(), nodesAdded+baseIdx);
        this->S[nodesAdded+baseIdx] = node[i].getS();
        nodesAdded += 1;
    }

    this->numNodes += nodesAdded;
}

__host__ 
void IbmNodesSoA::leftShiftNodesSoA(int idx, int left_shift){
    this->particleCenterIdx[idx-left_shift] = this->particleCenterIdx[idx];
    this->S[idx-left_shift] = this->S[idx];
    this->pos.leftShift(idx, left_shift);
    this->vel.leftShift(idx, left_shift);
    this->vel_old.leftShift(idx, left_shift);
    this->f.leftShift(idx, left_shift);
    this->deltaF.leftShift(idx, left_shift);
}


// #endif //PARTICLE_MODEL