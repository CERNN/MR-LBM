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

ParticleCenter* ParticlesSoA::getPCenterArray() const {
    return this->pCenterArray;
}

void ParticlesSoA::setPCenterArray(ParticleCenter* pArray) {
    pCenterArray = pArray;
}

dfloat3* ParticlesSoA::getPCenterLastPos() const {
    return pCenterLastPos;
}

void ParticlesSoA::setPCenterLastPos(dfloat3* pLastPos) {
    pCenterLastPos = pLastPos;
}

dfloat3* ParticlesSoA::getPCenterLastWPos() const {
    return pCenterLastWPos;
}

void ParticlesSoA::setPCenterLastWPos(dfloat3* pLastWPos) {
    pCenterLastWPos = pLastWPos;
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

    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        this->pCenterArray[p] = particles[p].getPCenter();
        this->pCenterLastPos[p] = particles[p].getPCenter().getPos();
        this->pCenterLastWPos[p] = particles[p].getPCenter().getW_old();
    }
    checkCudaErrors(cudaSetDevice(0));

}



#ifdef IBM


#endif // !IBM