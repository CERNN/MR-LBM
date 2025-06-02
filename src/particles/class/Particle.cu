#include "Particle.cuh"
#include <cstdlib>


Particle::Particle(){
    method = none; // Initialize method
}

ParticleMethod Particle::getMethod() const {return this->method;}
void Particle::setMethod(ParticleMethod method) { this->method = method;}

const ParticleCenter& Particle::getPCenter() const {return this->pCenter;}
void Particle::setPCenter(const ParticleCenter& pCenter) { this->pCenter = pCenter;}

const bool& Particle::getCollideParticle() const { return this->collideParticle; }
void Particle::setCollideParticle(const bool& value) { this->collideParticle = value; }

const bool& Particle::getCollideWall() const { return this->collideWall; }
void Particle::setCollideWall(const bool& value) { this->collideWall = value; }

ParticleShape Particle::getShape() const {return this->shape;}
void Particle::setShape(ParticleShape shape) { this->shape = shape;}


// ParticlesSoA class implementation
ParticlesSoA::ParticlesSoA() {
    pCenterArray = nullptr;
    pCenterLastPos = nullptr;
    pCenterLastWPos = nullptr;
}

ParticlesSoA::~ParticlesSoA() {
    // Free allocated memory
    if (pCenterArray != nullptr) {
        cudaFree(pCenterArray);
        pCenterArray = nullptr;
    }
    if (pCenterLastPos != nullptr) {
        free(pCenterLastPos);
        pCenterLastPos = nullptr;
    }
    if (pCenterLastWPos != nullptr) {
        free(pCenterLastWPos);
        pCenterLastWPos = nullptr;
    }
}

ParticleCenter* ParticlesSoA::getPCenterArray() const {return this->pCenterArray;}
void ParticlesSoA::setPCenterArray(ParticleCenter* pArray) {this->pCenterArray = pArray;}

dfloat3* ParticlesSoA::getPCenterLastPos() const {return this->pCenterLastPos;}
void ParticlesSoA::setPCenterLastPos(dfloat3* pLastPos) {this->pCenterLastPos = pLastPos;}

dfloat3* ParticlesSoA::getPCenterLastWPos() const {return this->pCenterLastWPos;}
void ParticlesSoA::setPCenterLastWPos(dfloat3* pLastWPos) {this->pCenterLastWPos = pLastWPos;}

ParticleShape* ParticlesSoA::getPShape() const {return this->pShape;}
void ParticlesSoA::setPShape(ParticleShape* pShape) {this->pShape = pShape;}

ParticleMethod* ParticlesSoA::getPMethod() const {return this->pMethod;}
void ParticlesSoA::setPMethod(ParticleMethod* pMethod) {this->pMethod = pMethod;}

bool* ParticlesSoA::getPCollideWall() const {return this->pCollideWall;}
void ParticlesSoA::setPCollideWall(bool* pMethod) {this->pCollideWall = pCollideWall;}

bool* ParticlesSoA::getPCollideParticle() const {return this->pCollideParticle;}
void ParticlesSoA::setPCollideParticle(bool* pMethod) {this->pCollideParticle = pCollideParticle;}

const MethodRange& ParticlesSoA::getMethodRange(ParticleMethod method) const {
    static const MethodRange empty{-1, -1};
    auto it = methodRanges.find(method);
    return (it != methodRanges.end()) ? it->second : empty;
}
void ParticlesSoA::setMethodRange(ParticleMethod method, int first, int last) {
    methodRanges[method] = {first, last};
}
int ParticlesSoA::getMethodCount(ParticleMethod method) const {
    MethodRange range = getMethodRange(method);
    if (range.first == -1 || range.last == -1 || range.last < range.first)
        return 0;  // No particles of this method
    return range.last - range.first + 1;
}


void ParticlesSoA::createParticles(Particle *particles){
   
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

void ParticlesSoA::updateParticlesAsSoA(Particle* particles){
    // Allocate particle center array
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->pCenterArray), sizeof(ParticleCenter) * NUM_PARTICLES));
    // Allocate array of last positions for Particles
    this->pCenterLastPos = (dfloat3*)malloc(sizeof(dfloat3)*NUM_PARTICLES);
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