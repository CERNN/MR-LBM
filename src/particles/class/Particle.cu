//#ifdef PARTICLE_MODEL

#include "particle.cuh"
#include <cstdlib>


__host__ __device__ Particle::Particle(){
    method = none; // Initialize method
}

__host__ __device__ ParticleMethod Particle::getMethod() const {return this->method;}
__host__ __device__ void Particle::setMethod(ParticleMethod method) { this->method = method;}

__host__ __device__ const ParticleCenter& Particle::getPCenter() const {return this->pCenter;}
__host__ __device__ void Particle::setPCenter(const ParticleCenter& pCenter) { this->pCenter = pCenter;}

__host__ __device__ const bool& Particle::getCollideParticle() const { return this->collideParticle; }
__host__ __device__ void Particle::setCollideParticle(const bool& value) { this->collideParticle = value; }

__host__ __device__ const bool& Particle::getCollideWall() const { return this->collideWall; }
__host__ __device__ void Particle::setCollideWall(const bool& value) { this->collideWall = value; }

__host__ __device__ ParticleShape Particle::getShape() const {return this->shape;}
__host__ __device__ void Particle::setShape(ParticleShape shape) { this->shape = shape;}


// ParticlesSoA class implementation
__host__ __device__
ParticlesSoA::ParticlesSoA() {
    pCenterArray = nullptr;
    pCenterLastPos = nullptr;
    pCenterLastWPos = nullptr;
    pShape = nullptr;
    pMethod = nullptr;
    pCollideWall = nullptr;
    pCollideParticle = nullptr;
}

__host__ __device__
ParticlesSoA::~ParticlesSoA() {
    if (pCenterArray) {
        cudaFree(pCenterArray);
        pCenterArray = nullptr;
    }
    if (pCenterLastPos) {
        free(pCenterLastPos);
        pCenterLastPos = nullptr;
    }
    if (pCenterLastWPos) {
        free(pCenterLastWPos);
        pCenterLastWPos = nullptr;
    }
    if (pShape) {
        free(pShape);
        pShape = nullptr;
    }
    if (pMethod) {
        free(pMethod);
        pMethod = nullptr;
    }
    if (pCollideWall) {
        free(pCollideWall);
        pCollideWall = nullptr;
    }
    if (pCollideParticle) {
        free(pCollideParticle);
        pCollideParticle = nullptr;
    }
}

__host__ __device__ ParticleCenter* ParticlesSoA::getPCenterArray() const {return this->pCenterArray;}
__host__ __device__ void ParticlesSoA::setPCenterArray(ParticleCenter* pArray) {this->pCenterArray = pArray;}

__host__ __device__ dfloat3* ParticlesSoA::getPCenterLastPos() const {return this->pCenterLastPos;}
__host__ __device__ void ParticlesSoA::setPCenterLastPos(dfloat3* pLastPos) {this->pCenterLastPos = pLastPos;}

__host__ __device__ dfloat3* ParticlesSoA::getPCenterLastWPos() const {return this->pCenterLastWPos;}
__host__ __device__ void ParticlesSoA::setPCenterLastWPos(dfloat3* pLastWPos) {this->pCenterLastWPos = pLastWPos;}

__host__ __device__ ParticleShape* ParticlesSoA::getPShape() const {return this->pShape;}
__host__ __device__ void ParticlesSoA::setPShape(ParticleShape* pShape) {this->pShape = pShape;}

__host__ __device__ ParticleMethod* ParticlesSoA::getPMethod() const {return this->pMethod;}
__host__ __device__ void ParticlesSoA::setPMethod(ParticleMethod* pMethod) {this->pMethod = pMethod;}

__host__ __device__ bool* ParticlesSoA::getPCollideWall() const {return this->pCollideWall;}
__host__ __device__ void ParticlesSoA::setPCollideWall(bool* pMethod) {this->pCollideWall = pCollideWall;}

__host__ __device__ bool* ParticlesSoA::getPCollideParticle() const {return this->pCollideParticle;}
__host__ __device__ void ParticlesSoA::setPCollideParticle(bool* pMethod) {this->pCollideParticle = pCollideParticle;}

__device__ __host__
const MethodRange& ParticlesSoA::getMethodRange(ParticleMethod method) const {
    static const MethodRange empty{-1, -1};
    auto it = methodRanges.find(method);
    return (it != methodRanges.end()) ? it->second : empty;
}

__device__ __host__
void ParticlesSoA::setMethodRange(ParticleMethod method, int first, int last) {
    methodRanges[method] = {first, last};
}

__device__ __host__
int ParticlesSoA::getMethodCount(ParticleMethod method) const {
    MethodRange range = getMethodRange(method);
    if (range.first == -1 || range.last == -1 || range.last < range.first)
        return 0;  // No particles of this method
    return range.last - range.first + 1;
}


__host__ __device__ void ParticlesSoA::createParticles(Particle *particles){
   
    #include CASE_PARTICLE_CREATE

    for(int i = 0; i <NUM_PARTICLES ; i++){

        switch (particles[i].getMethod())
        {
        case PIBM:
            /* code */
            break;
        case IBM:
            //particles[i].makeSpherePolar(PARTICLE_DIAMETER, center[i], MESH_COULOMB, true, PARTICLE_DENSITY, vel, w);
          //  particles[i].makeEllipsoid(dfloat3(40.0,20.0,10.0), center[i], dfloat3(0.5,1.0,0.6), 0.3*M_PI/4,true, PARTICLE_DENSITY, vel, w);
            //particles[i].makeCapsule(PARTICLE_DIAMETER, center1, center2, true,PARTICLE_DENSITY, vel, w);
            break;
        case TRACER:
            break;
        default:
            break;
        }
        
    }
}

__host__ __device__ void ParticlesSoA::updateParticlesAsSoA(Particle* particles){
    // Allocate particle center array
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->pCenterArray), sizeof(ParticleCenter) * NUM_PARTICLES));
    // Allocate array of last positions for Particles
    this->pCenterLastPos = (dfloat3*)malloc(sizeof(dfloat3) * NUM_PARTICLES);
    this->pCenterLastWPos = (dfloat3*)malloc(sizeof(dfloat3) * NUM_PARTICLES);

    checkCudaErrors(cudaSetDevice(0));

    int particleCount = 0;
    std::map<ParticleMethod, bool> methodSeen;
    
    //LAMBDA FUNCTION
    auto insertByMethod = [&](ParticleMethod method) {
        int firstIndex = particleCount;
        for (int p = 0; p < NUM_PARTICLES; ++p) {
            if (particles[p].getMethod() == method) {
                // Insert particle data into SoA arrays
                this->pCenterArray[particleCount]       = particles[p].getPCenter();
                this->pCenterLastPos[particleCount]     = particles[p].getPCenter().getPos();
                this->pCenterLastWPos[particleCount]    = particles[p].getPCenter().getW_old();
                this->pShape[particleCount]             = particles[p].getShape();
                this->pMethod[particleCount]            = particles[p].getMethod();
                this->pCollideWall[particleCount]       = particles[p].getCollideWall();
                this->pCollideParticle[particleCount]   = particles[p].getCollideParticle();

                ++particleCount;
            }
        }
        if (particleCount > firstIndex) {  // Only set if any particles of this type were found
            int lastIndex = particleCount - 1;
            this->setMethodRange(method, firstIndex, lastIndex);
        }
    };

    insertByMethod(IBM);
    insertByMethod(PIBM);
    insertByMethod(TRACER);


  


    checkCudaErrors(cudaSetDevice(0));

}



#ifdef IBM
#endif // !IBM


//#endif //PARTICLE_MODEL