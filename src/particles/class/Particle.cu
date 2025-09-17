#include "particle.cuh"
#include <cstdlib>
#include <iostream>

#ifdef PARTICLE_MODEL

__host__ __device__ Particle::Particle(){
    method = none; // Initialize method
    numNodes = 0; // Initialize numNodes
    nodes = nullptr; // Initialize nodes
}

__host__ Particle::~Particle(){
    if (pCenter) {
        delete pCenter;
        pCenter = nullptr;
    }
    if (shape) {
        delete shape;
        shape = nullptr;
    }
    if (nodes) {
        delete nodes;
        nodes = nullptr;
    }
}

__host__ __device__ unsigned int Particle::getNumNodes() const {return this->numNodes;}
__host__ __device__ void Particle::setNumNodes(unsigned int numNodes) {this->numNodes = numNodes;}

__host__ __device__ IbmNodes* Particle::getNode() const {return this->nodes;}
__host__ __device__ void Particle::setNode(IbmNodes* nodes) { this->nodes = nodes;}

__host__ __device__ ParticleMethod Particle::getMethod() const {return this->method;}
__host__ __device__ void Particle::setMethod(ParticleMethod method) { this->method = method;}

__host__ __device__ ParticleCenter* Particle::getPCenter() const {return this->pCenter;}
__host__ __device__ void Particle::setPCenter(ParticleCenter* pCenter) { this->pCenter = pCenter;}

__host__ __device__ const bool& Particle::getCollideParticle() const { return this->collideParticle; }
__host__ __device__ void Particle::setCollideParticle(const bool& value) { this->collideParticle = value; }

__host__ __device__ const bool& Particle::getCollideWall() const { return this->collideWall; }
__host__ __device__ void Particle::setCollideWall(const bool& value) { this->collideWall = value; }

__host__ __device__ ParticleShape* Particle::getShape() const {return this->shape;}
__host__ __device__ void Particle::setShape(ParticleShape* shape) { this->shape = shape;}

// ParticlesSoA class implementation
__host__ 
ParticlesSoA::ParticlesSoA() {
    pCenterArray = nullptr;
    pCenterLastPos = nullptr;
    pCenterLastWPos = nullptr;
    pShape = nullptr;
    pMethod = nullptr;
    pCollideWall = nullptr;
    pCollideParticle = nullptr;
}

__host__ 
ParticlesSoA::~ParticlesSoA() {
    if (pCenterArray) {
        cudaFree(pCenterArray);
        pCenterArray = nullptr;
    }
    if (pCenterLastPos) {
        cudaFree(pCenterLastPos);
        pCenterLastPos = nullptr;
    }
    if (pCenterLastWPos) {
        cudaFree(pCenterLastWPos);
        pCenterLastWPos = nullptr;
    }
    if (pShape) {
        cudaFree(pShape);
        pShape = nullptr;
    }
    if (pMethod) {
        cudaFree(pMethod);
        pMethod = nullptr;
    }
    if (pCollideWall) {
        cudaFree(pCollideWall);
        pCollideWall = nullptr;
    }
    if (pCollideParticle) {
        cudaFree(pCollideParticle);
        pCollideParticle = nullptr;
    }
}

__host__ __device__ IbmNodesSoA* ParticlesSoA::getNodesSoA() {
    return this->nodesSoA;
}

__host__ __device__ void ParticlesSoA::setNodesSoA(const IbmNodesSoA* nodes) {
    for (int i = 0; i < N_GPUS; ++i) {
        this->nodesSoA[i] = nodes[i];  // copia dos elementos
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

__host__
const MethodRange& ParticlesSoA::getMethodRange(ParticleMethod method) const {
    static const MethodRange empty{-1, -1};
    auto it = methodRanges.find(method);
    return (it != methodRanges.end()) ? it->second : empty;
}

__host__
void ParticlesSoA::setMethodRange(ParticleMethod method, int first, int last) {
    methodRanges[method] = {first, last};
}

__host__
int ParticlesSoA::getMethodCount(ParticleMethod method) const {
    MethodRange range = getMethodRange(method);
    if (range.first == -1 || range.last == -1 || range.last < range.first)
        return 0;  // No particles of this method
    return range.last - range.first + 1;
}

__host__ void ParticlesSoA::createParticles(Particle *particles){
   
    centerStorage.resize(NUM_PARTICLES);
    pShape = new ParticleShape[NUM_PARTICLES];

    #include CASE_PARTICLE_CREATE

    if (pShape == nullptr) {
        pShape = new ParticleShape[NUM_PARTICLES]; 
        for (int i = 0; i < NUM_PARTICLES; i++) {
            pShape[i] = SPHERE;
            particles[i].setShape(&pShape[i]);
        }
    }


    for(int i = 0; i <NUM_PARTICLES ; i++){

        switch (particles[i].getMethod())
        {
        case PIBM:
            break;
        case IBM:
            switch (*particles[i].getShape())
            {
            case SPHERE:
                particles[i].makeSpherePolar(particles[i].getPCenter());
                break;
            case CAPSULE:
                particles[i].makeCapsule(particles[i].getPCenter());
                break;
            case ELLIPSOID:
               particles[i].makeEllipsoid(particles[i].getPCenter());
                break;
            case GRID:
                particles[i].makeUniformBox(particles[i].getPCenter());                
                break;
            case RANDOM:
                particles[i].makeRandomBox(particles[i].getPCenter());
                break;
            default:
                break;
            }
            break;
        case TRACER:
            break;
        default:
            break;
        }
        
    }
}

__host__ void ParticlesSoA::updateParticlesAsSoA(Particle* particles){

    if (particles == nullptr) {
        printf("ERROR: particles is nullptr!\n\n"); fflush(stdout);
        return;
    }
    if (NUM_PARTICLES <= 0) {
        printf("ERROR: Invalid NUM_PARTICLES!\n\n"); fflush(stdout);
        return;
    }

    checkCudaErrors(cudaSetDevice(0));

    unsigned int totalIbmNodes = 0;

    // Determine the total number of nodes
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        totalIbmNodes += particles[p].getNumNodes();
    }

    printf("Total number of nodes: %u\n", totalIbmNodes);
    printf("Total memory used for Particles: %lu Mb\n",
           (unsigned long)((totalIbmNodes * sizeof(IbmNodes) * N_GPUS + NUM_PARTICLES * sizeof(ParticleCenter)) / BYTES_PER_MB));
    fflush(stdout);

    printf("Allocating particles in GPU... \t"); fflush(stdout);

    this->nodesSoA[0].allocateMemory(totalIbmNodes);

    printf("Success \n"); fflush(stdout);

    checkCudaErrors(cudaMallocManaged((void**)&this->pCenterArray,       sizeof(ParticleCenter) * NUM_PARTICLES));
    checkCudaErrors(cudaMallocManaged((void**)&this->pCenterLastPos,     sizeof(dfloat3)        * NUM_PARTICLES));
    checkCudaErrors(cudaMallocManaged((void**)&this->pCenterLastWPos,    sizeof(dfloat3)        * NUM_PARTICLES));
    checkCudaErrors(cudaMallocManaged((void**)&this->pShape,             sizeof(ParticleShape)  * NUM_PARTICLES));
    checkCudaErrors(cudaMallocManaged((void**)&this->pMethod,            sizeof(ParticleMethod) * NUM_PARTICLES));
    checkCudaErrors(cudaMallocManaged((void**)&this->pCollideWall,       sizeof(bool)           * NUM_PARTICLES));
    checkCudaErrors(cudaMallocManaged((void**)&this->pCollideParticle,   sizeof(bool)           * NUM_PARTICLES));

    if (!this->pCenterArray || !pCenterLastPos || !pCenterLastWPos ||
        !this->pShape || !this->pMethod || !this->pCollideWall || !this->pCollideParticle) {
        printf("ERRO: Memory allocation failed!!\n"); fflush(stdout);
        return;
    }

    auto insertByMethod = [&](ParticleMethod method) {
        int firstIndex = -1;
        int lastIndex = -1;

        for (int p = 0; p < NUM_PARTICLES; ++p) {     
            if (particles[p].getMethod() != method)
                continue;
            ParticleCenter* pc = particles[p].getPCenter();

            if (!pc) {
                printf("NOTICE: Particle %d with pc == nullptr\n", p); fflush(stdout);
                continue;
            }

            this->pCenterArray[p]       = *pc;
            this->pCenterLastPos[p]     = pc->getPos_old();
            this->pCenterLastWPos[p]    = pc->getW_old();
            this->pShape[p]             = *(particles[p].getShape());
            this->pMethod[p]            = particles[p].getMethod();
            this->pCollideWall[p]       = particles[p].getCollideWall();
            this->pCollideParticle[p]   = particles[p].getCollideParticle();
            this->nodesSoA[0].copyNodesFromParticle(&particles[p], p, 0);
            if (firstIndex == -1) firstIndex = p;
            lastIndex = p;
        }
       if (firstIndex != -1) {
              this->setMethodRange(method, firstIndex, lastIndex);
        }
     };

    insertByMethod(IBM);
    insertByMethod(PIBM);
    insertByMethod(TRACER);
}

void ParticlesSoA::freeNodesAndCenters(){
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        this->nodesSoA[i].freeMemory();
    }
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    cudaFree(this->pCenterArray);
    this->pCenterArray = nullptr;
    cudaFree(this->pCenterLastPos);
    this->pCenterLastPos = nullptr;
    cudaFree(this->pCenterLastWPos);
    this->pCenterLastWPos = nullptr;
    cudaFree(this->pShape);
    this->pShape = nullptr;
    cudaFree(this->pMethod);
    this->pMethod = nullptr;
    cudaFree(this->pCollideWall);
    this->pCollideWall = nullptr;
    cudaFree(this->pCollideParticle);
    this->pCollideParticle = nullptr;
}

#ifdef IBM_METHOD


__host__
void Particle::makeUniformBox(ParticleCenter *particleCenter)
{
    // Particle density
    pCenter->setDensity(2.0);
    // Particle center position
    pCenter->setPos({(NX-1)/2.0,(NY-1)/2.0,(NZ-1)/2.0});
    pCenter->setPos_old({(NX-1)/2.0,(NY-1)/2.0,(NZ-1)/2.0});
    // Particle velocity
    pCenter->setVel({0,0,0});
    pCenter->setVel_old({0,0,0});
    // Particle rotation
    pCenter->setW({0,0,0});
    pCenter->setW_avg({0,0,0});
    pCenter->setW_old({0,0,0});
    //
    pCenter->setQPosW(0);
    pCenter->setQPosX(1);
    pCenter->setQPosY(0);
    pCenter->setQPosZ(0);

    pCenter->setQPosOldW(0.0);
    pCenter->setQPosOldX(1.0);
    pCenter->setQPosOldY(0.0);
    pCenter->setQPosOldZ(0.0);

    // Innertia momentum
    pCenter->setIXX(1.0);
    pCenter->setIYY(1.0);
    pCenter->setIZZ(1.0);

    pCenter->setIXY(0.0);
    pCenter->setIXZ(0.0);
    pCenter->setIYZ(0.0);

    pCenter->setFX(0.0);
    pCenter->setFY(0.0);
    pCenter->setFZ(0.0);

    pCenter->setFOldX(0.0);
    pCenter->setFOldY(0.0);
    pCenter->setFOldZ(0.0);

    pCenter->setMX(0.0);
    pCenter->setMY(0.0);
    pCenter->setMZ(0.0);

    pCenter->setMOldX(0.0);
    pCenter->setMOldY(0.0);
    pCenter->setMOldZ(0.0);

    pCenter->setMovable(false);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    int Npoints = particleCenter->getDiameter(); //USING DIAMTER TO PASS THE NUMBER OF NODES

    // estimate number of points per axis
    unsigned int n = (unsigned int) round(cbrt((dfloat)Npoints));
    
    // spacing
    dfloat dx = (dfloat)(NX-1) / n;
    dfloat dy = (dfloat)(NY-1) / n;
    dfloat dz = (dfloat)(NZ-1) / n;

    int Nx = n;
    int Ny = n;
    int Nz = n;
    this->numNodes = Nx * Ny * Nz;
    this->nodes = (IbmNodes*) malloc(sizeof(IbmNodes) * this->numNodes);

    unsigned int nodeIndex = 0;

    for (unsigned int i = 0; i < Nx; i++) {
        for (unsigned int j = 0; j < Ny; j++) {
            for (unsigned int k = 0; k < Nz; k++) {
                dfloat3 pos_node;
                pos_node.x = (i * dx) + dx/2.0;
                pos_node.y = (j * dy) + dy/2.0;
                pos_node.z = (k * dz) + dz/2.0;

                this->nodes[nodeIndex].setPos(pos_node);
                dfloat node_s = 1.0;
                this->nodes[nodeIndex].setS(node_s);

                nodeIndex++;
            }
        }
    }
}

__host__
void Particle::makeRandomBox(ParticleCenter *particleCenter)
{

    // Particle density
    pCenter->setDensity(particleCenter->getDensity());
    // Particle center position
    pCenter->setPos({(NX-1)/2.0,(NY-1)/2.0,(NZ-1)/2.0});
    pCenter->setPos_old({(NX-1)/2.0,(NY-1)/2.0,(NZ-1)/2.0});
    // Particle velocity
    pCenter->setVel({0,0,0});
    pCenter->setVel_old({0,0,0});
    // Particle rotation
    pCenter->setW({0,0,0});
    pCenter->setW_avg({0,0,0});
    pCenter->setW_old({0,0,0});
    //
    pCenter->setQPosW(0);
    pCenter->setQPosX(1);
    pCenter->setQPosY(0);
    pCenter->setQPosZ(0);

    pCenter->setQPosOldW(0.0);
    pCenter->setQPosOldX(1.0);
    pCenter->setQPosOldY(0.0);
    pCenter->setQPosOldZ(0.0);

    // Innertia momentum
    pCenter->setIXX(1.0);
    pCenter->setIYY(1.0);
    pCenter->setIZZ(1.0);

    pCenter->setIXY(0.0);
    pCenter->setIXZ(0.0);
    pCenter->setIYZ(0.0);

    pCenter->setFX(0.0);
    pCenter->setFY(0.0);
    pCenter->setFZ(0.0);

    pCenter->setFOldX(0.0);
    pCenter->setFOldY(0.0);
    pCenter->setFOldZ(0.0);

    pCenter->setMX(0.0);
    pCenter->setMY(0.0);
    pCenter->setMZ(0.0);

    pCenter->setMOldX(0.0);
    pCenter->setMOldY(0.0);
    pCenter->setMOldZ(0.0);

    pCenter->setMovable(false);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    int Npoints = particleCenter->getDiameter(); // USING DIAMETER TO PASS THE NUMBER OF NODES

    this->numNodes = Npoints;
    this->nodes = (IbmNodes*) malloc(sizeof(IbmNodes) * this->numNodes);

    unsigned int nodeIndex = 0;

    for (unsigned int n = 0; n < Npoints; n++) {
        dfloat3 pos_node;

        // random numbers between 0 and 1, scaled to domain size
        pos_node.x = ((dfloat) rand() / (dfloat) RAND_MAX) * (NX - 1);
        pos_node.y = ((dfloat) rand() / (dfloat) RAND_MAX) * (NY - 1);
        pos_node.z = ((dfloat) rand() / (dfloat) RAND_MAX) * (NZ - 1);

        this->nodes[nodeIndex].setPos(pos_node);

        dfloat node_s = 1.0; // uniform weight
        this->nodes[nodeIndex].setS(node_s);

        nodeIndex++;
    }
}

// dfloat diameter, dfloat3 center, unsigned int coulomb, bool move,dfloat density, dfloat3 vel, dfloat3 w
__host__
void Particle::makeSpherePolar(ParticleCenter *particleCenter)
{
    // Maximum number of layer of sphere
    //unsigned int maxNumLayers = 5000;
    // Number of layers in sphere
    unsigned int nLayer;
    // Number of nodes per layer in sphere
    unsigned int* nNodesLayer;
    // Angles in polar coordinates and node area
    dfloat *theta, *zeta, *S;

    dfloat phase = 0.0;

    // Define the properties of the particle
    dfloat r = particleCenter->getDiameter() / 2.0;
    dfloat volume = r*r*r*4*M_PI/3;

    pCenter->setRadius(r);
    pCenter->setVolume(r*r*r*4*M_PI/3);

    // Particle area
    pCenter->setS( 4.0 * M_PI * r * r);

    // Particle density
    pCenter->setDensity(particleCenter->getDensity());

    // Particle center position
    pCenter->setPos(particleCenter->getPos());
    pCenter->setPos_old(particleCenter->getPos_old());

    // Particle velocity
    pCenter->setVel(particleCenter->getVel());
    pCenter->setVel_old(particleCenter->getVel_old());

    // Particle rotation
    pCenter->setW(particleCenter->getW());
    pCenter->setW_avg(particleCenter->getW_avg());
    pCenter->setW_old(particleCenter->getW_old());

    pCenter->setQPosW(0);
    pCenter->setQPosX(1);
    pCenter->setQPosY(0);
    pCenter->setQPosZ(0);

    pCenter->setQPosOldW(0.0);
    pCenter->setQPosOldX(1.0);
    pCenter->setQPosOldY(0.0);
    pCenter->setQPosOldZ(0.0);
    
    // Innertia momentum
    pCenter->setIXX(2.0 * volume * pCenter->getDensity() * r * r / 5.0);
    pCenter->setIYY(2.0 * volume * pCenter->getDensity() * r * r / 5.0);
    pCenter->setIZZ(2.0 * volume * pCenter->getDensity() * r * r / 5.0);

    pCenter->setIXY(0.0);
    pCenter->setIXZ(0.0);
    pCenter->setIYZ(0.0);

    pCenter->setFX(0.0);
    pCenter->setFY(0.0);
    pCenter->setFZ(0.0);

    pCenter->setFOldX(0.0);
    pCenter->setFOldY(0.0);
    pCenter->setFOldZ(0.0);

    pCenter->setMX(0.0);
    pCenter->setMY(0.0);
    pCenter->setMZ(0.0);

    pCenter->setMOldX(0.0);
    pCenter->setMOldY(0.0);
    pCenter->setMOldZ(0.0);

    pCenter->setMovable(particleCenter->getMovable());

    pCenter->setSemiAxis1(dfloat3(r,r,r));

    // for(int i = 0; i <MAX_ACTIVE_COLLISIONS;i++){
    //     this->pCenter.collision.collisionPartnerIDs[i] = -1;
    //     this->pCenter.collision.tangentialDisplacements[i] = dfloat3(0,0,0);
    //     this->pCenter.collision.lastCollisionStep[i] = -1;
    // }
    
    pCenter->getCollision().reset(); 

    // for (int i = 0; i < MAX_ACTIVE_COLLISIONS; ++i) {
    //     dfloat3 displacement = pCenter->getCollision().getTangentialDisplacement(i);
    // }

    //breugem correction
    r -= BREUGEM_PARAMETER;

    // Number of layers in the sphere
    nLayer = (unsigned int)(2.0 * sqrt(2) * r / MESH_SCALE + 1.0); 

    nNodesLayer = (unsigned int*)malloc((nLayer+1) * sizeof(unsigned int));
    theta = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));
    zeta = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));
    S = (dfloat*)malloc((nLayer+1) * sizeof(dfloat));

    this->numNodes = 0;
    for (int i = 0; i <= nLayer; i++) {
        // Angle of each layer
        theta[i] = M_PI * ((dfloat)i / (dfloat)nLayer - 0.5); 
        // Determine the number of node per layer
        nNodesLayer[i] = (unsigned int)(1.5 + cos(theta[i]) * nLayer * sqrt(3)); 
        // Total number of nodes on the sphere
        this->numNodes += nNodesLayer[i]; 
        zeta[i] = r * sin(theta[i]); // Height of each layer
    }

    
    for (int i = 0; i < nLayer; i++) {
        // Calculate the distance to the south pole to the mid distance of the layer and previous layer
        S[i] = (zeta[i] + zeta[i + 1]) / 2.0 - zeta[0]; 
    }
    S[nLayer] = 2 * r;
    for (int i = 0; i <= nLayer; i++) {
        // Calculate the area of sphere segment since the south pole
        S[i] = 2 * M_PI * r * S[i]; 
    }
    for (int i = nLayer; i > 0; i--) {
        // Calculate the area of the layer
        S[i] = S[i] - S[i - 1];
    }
    S[0] = S[nLayer];
    

    this->nodes = (IbmNodes*) malloc(sizeof(IbmNodes) * this->numNodes);

    IbmNodes* first_node = &(this->nodes[0]);

    // South node - define all properties
    first_node->setPos(dfloat3(0, 0, r * sin(theta[0])));

    dfloat3 pos_node = first_node->getPos();

    first_node->setS(S[0]);
    dfloat node_s = first_node->getS();

    // Define node velocity
    dfloat3 newVel;

    newVel.x = particleCenter->getVel().x + particleCenter->getW().y * pos_node.z - particleCenter->getW().z * pos_node.y;
    newVel.y = particleCenter->getVel().y + particleCenter->getW().z * pos_node.x - particleCenter->getW().x * pos_node.z;
    newVel.z = particleCenter->getVel().z + particleCenter->getW().x * pos_node.y - particleCenter->getW().y * pos_node.x;

    first_node->setVel(newVel);
    first_node->setVelOld(newVel);
    
    int nodeIndex = 1;
    for (int i = 1; i < nLayer; i++) {
        if (i % 2 == 1) {
            // Calculate the phase of the segmente to avoid a straight point line
            phase = phase + M_PI / nNodesLayer[i];
        }

        for (int j = 0; j < nNodesLayer[i]; j++) {
            // Determine the properties of each node in the mid layers
            dfloat3 pos_nodeslayer;
            pos_nodeslayer.x = r * cos(theta[i]) * cos((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            pos_nodeslayer.y = r * cos(theta[i]) * sin((dfloat)j * 2.0 * M_PI / nNodesLayer[i] + phase);
            pos_nodeslayer.z = r * sin(theta[i]);

            this->nodes[nodeIndex].setPos(pos_nodeslayer);
            // The area of sphere segment is divided by the number of node
            // in the layer, so all nodes have the same area in the layer
            this->nodes[nodeIndex].setS(S[i] / nNodesLayer[i]);

            // Define node velocity
            dfloat3 vel_nodeslayer;

            vel_nodeslayer.x = particleCenter->getVel().x + particleCenter->getW().y * pos_nodeslayer.z - particleCenter->getW().z * pos_nodeslayer.y;
            vel_nodeslayer.y = particleCenter->getVel().y + particleCenter->getW().z * pos_nodeslayer.x - particleCenter->getW().x * pos_nodeslayer.z;
            vel_nodeslayer.z = particleCenter->getVel().z + particleCenter->getW().x * pos_nodeslayer.y - particleCenter->getW().y *pos_nodeslayer.x;

            this->nodes[nodeIndex].setVel(vel_nodeslayer);
            this->nodes[nodeIndex].setVelOld(vel_nodeslayer);

            // Add one node
            nodeIndex++;
        }
    }

    // North pole -define all properties
    IbmNodes* last_node = &(this->nodes[this->numNodes-1]);

    dfloat3 pos_last_node;
    pos_last_node.x = 0;
    pos_last_node.y = 0;
    pos_last_node.z = r * sin(theta[nLayer]);
    last_node->setPos(pos_last_node);

    last_node->setS(S[nLayer]);

    dfloat3 pos_last_node_get = last_node->getPos();
    dfloat3 vel_last_node;
    vel_last_node.x = particleCenter->getVel().x + particleCenter->getW().y * pos_last_node_get.z - particleCenter->getW().z * pos_last_node_get.y;
    vel_last_node.y = particleCenter->getVel().y + particleCenter->getW().z * pos_last_node_get.x - particleCenter->getW().x * pos_last_node_get.z;
    vel_last_node.z = particleCenter->getVel().z + particleCenter->getW().x * pos_last_node_get.y - particleCenter->getW().y * pos_last_node_get.x;
    last_node->setVel(vel_last_node);
    last_node->setVelOld(vel_last_node);

    unsigned int numNodes = this->numNodes;

    IbmNodes* node_i;
    // Coulomb node positions distribution
     if (MESH_COULOMB != 0) {
         dfloat3 dir;
         dfloat mag;
         dfloat scaleF;
         dfloat3* cForce;
         cForce = (dfloat3*)malloc(numNodes * sizeof(dfloat3));
        IbmNodes* node_j;

         scaleF = 0.1;
         dfloat fx, fy, fz;
         for (unsigned int c = 0; c < MESH_COULOMB; c++) {
             for (int i = 0; i < numNodes; i++) {
                 cForce[i].x = 0;
                 cForce[i].y = 0;
                 cForce[i].z = 0;
             }
             for (int i = 0; i < numNodes; i++) {
                 node_i = &(this->nodes[i]);
                 for (int j = i+1; j < numNodes; j++) {
                     IbmNodes* node_j = &(this->nodes[j]);
                     dir.x = node_j->getPosX() - node_i->getPosX();
                     dir.y = node_j->getPosY() - node_i->getPosY();
                     dir.z = node_j->getPosZ() - node_i->getPosZ();
                     mag = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
                     cForce[i].x -= dir.x / mag;
                     cForce[i].y -= dir.y / mag;
                     cForce[i].z -= dir.z / mag;
                     cForce[j].x -= -dir.x / mag;
                     cForce[j].y -= -dir.y / mag;
                     cForce[j].z -= -dir.z / mag;
                 }
             }
             for (int i = 0; i < numNodes; i++) {
                 // Move particle
                 fx = cForce[i].x / scaleF;
                 fy = cForce[i].y / scaleF;
                 fz = cForce[i].z / scaleF;          
                 IbmNodes* node_i = &(this->nodes[i]);
                 node_i->setPosX(node_i->getPosX() + fx);
                 node_i->setPosY(node_i->getPosY() + fy);
                 node_i->setPosZ(node_i->getPosZ() + fz);
                 // Return to sphere
                 mag = sqrt(node_i->getPosX() * node_i->getPosX() 
                          + node_i->getPosY() * node_i->getPosY() 
                          + node_i->getPosZ() * node_i->getPosZ());
                 node_i->setPosX(node_i->getPosX() * r / mag);
                 node_i->setPosY(node_i->getPosY() * r / mag);
                 node_i->setPosZ(node_i->getPosZ() * r / mag);                          
             }
         }
         // Area fix
         for (int i = 0; i < numNodes; i++) {
             IbmNodes* node_i = &(this->nodes[i]);
             node_i->setS(pCenter->getS()/ (numNodes));
         }
         // Free coulomb force
         free(cForce);
         dfloat dA = pCenter->getS()/ (numNodes);
         for(int i = 0; i < numNodes; i++){
             IbmNodes* node_i = &(this->nodes[i]);
             node_i->setS(dA);
         }
     }

    for (int i = 0; i < numNodes; i++) {
        node_i = &(this->nodes[i]);
    
        dfloat3 pos_node_i = node_i->getPos();
        dfloat3 center_node_i = particleCenter->getPos();

        pos_node_i.x += center_node_i.x;
        pos_node_i.y += center_node_i.y;
        pos_node_i.z += center_node_i.z;
    
        node_i->setPos(pos_node_i);

    }

    // for(int ii = 0;ii<numNodes;ii++){
    //     IbmNodes* node_j = &(this->nodes[ii]);
    //     dfloat3 updatedPos = node_j->getPos();
    // }
   
    // Free allocated variables
    free(nNodesLayer);
    free(theta);
    free(zeta);
    free(S);

    // Update old position value

    pCenter->setPos_old(pCenter->getPos());
    
}

void Particle::makeCapsule(ParticleCenter *particleCenter){

    //radius
    dfloat r = particleCenter->getDiameter()/2.00;

    //length cylinder
    dfloat length = sqrt((particleCenter->getSemiAxis2().x - particleCenter->getSemiAxis1().x)*(particleCenter->getSemiAxis2().x - particleCenter->getSemiAxis1().x) + 
                         (particleCenter->getSemiAxis2().y - particleCenter->getSemiAxis1().y)*(particleCenter->getSemiAxis2().y - particleCenter->getSemiAxis1().y) + 
                         (particleCenter->getSemiAxis2().z - particleCenter->getSemiAxis1().z)*(particleCenter->getSemiAxis2().z - particleCenter->getSemiAxis1().z));
    
    //unit vector of capsule direction
    dfloat3 vec = dfloat3((particleCenter->getSemiAxis2().x - particleCenter->getSemiAxis1().x)/length,
                          (particleCenter->getSemiAxis2().y - particleCenter->getSemiAxis1().y)/length,
                          (particleCenter->getSemiAxis2().z - particleCenter->getSemiAxis1().z)/length);

    //number of slices
    int nSlices = (int)(length/MESH_SCALE);
    //array location of height of slices
    dfloat* z_cylinder = (dfloat*)malloc(nSlices * sizeof(dfloat));
    for (int i = 0; i < nSlices; i++)
        z_cylinder[i] = length * i / (nSlices - 1);

    // Calculate the number of nodes in each circle
    int nCirclePoints = (int)(2 * M_PI * r / MESH_SCALE);
    //array location of theta
    dfloat* theta = (dfloat*)malloc(nCirclePoints * sizeof(dfloat));
    for (int i = 0; i < nCirclePoints; i++)
        theta[i] = 2 * M_PI * i / (nCirclePoints - 1);

    // Cylinder surface nodes
    dfloat3* cylinder = (dfloat3*)malloc(nSlices * nCirclePoints * sizeof(dfloat3));

    for (int i = 0; i < nSlices; i++) {
        for (int j = 0; j < nCirclePoints; j++) {
            cylinder[i * nCirclePoints + j].x = r * cos(theta[j]);
            cylinder[i * nCirclePoints + j].y = r * sin(theta[j]);
            cylinder[i * nCirclePoints + j].z = z_cylinder[i];
        }
    }

    //spherical caps
    //calculate the number of slices and the angle of the slices
    int nSlices_sphere = (int)(M_PI * r / (2 * MESH_SCALE));
    dfloat* phi = (dfloat*)malloc(nSlices_sphere * sizeof(dfloat));
    for (int i = 0; i < nSlices_sphere; i++)
        phi[i] = (M_PI / 2) * i / (nSlices_sphere - 1);

    //calculate the number of nodes in the cap
    int cap_count = 0;
    for (int i = 0; i < nSlices_sphere; i++) {
        dfloat rSlice = r * cos(phi[i]);
        int nCirclePoints_sphere = (int)(2 * M_PI * rSlice / MESH_SCALE);
        for (int j = 0; j < nCirclePoints_sphere; j++) 
            cap_count++;
    }

    dfloat3* cap1 = (dfloat3*)malloc(cap_count * sizeof(dfloat3));
    dfloat3* cap2 = (dfloat3*)malloc(cap_count * sizeof(dfloat3));

    cap_count = 0;
    dfloat Xs, Ys, Zs;
    for (int i = 0; i < nSlices_sphere; i++) {
        dfloat rSlice = r * cos(phi[i]);
        int nCirclePoints_sphere = floor(2 * M_PI * rSlice / MESH_SCALE);
        
        for (int j = 0; j < nCirclePoints_sphere; j++)
            theta[j] = 2 * M_PI * j / (nCirclePoints_sphere - 1);


        for (int j = 0; j < nCirclePoints_sphere; j++) {
            Xs = rSlice * cos(theta[j]);
            Ys = rSlice * sin(theta[j]);
            Zs = r * sin(phi[i]);

            cap1[cap_count].x = Xs;
            cap1[cap_count].y = Ys;
            cap1[cap_count].z = -Zs;

            cap2[cap_count].x = Xs;
            cap2[cap_count].y = Ys;
            cap2[cap_count].z = Zs + length;
            cap_count++;
        }
    }

    dfloat3* cap1_filtered = (dfloat3*)malloc(cap_count * sizeof(dfloat3));
    dfloat3* cap2_filtered = (dfloat3*)malloc(cap_count * sizeof(dfloat3));

    //remove the nodes from inside, if necessary

    int cap_filtered_count = 0;
    for (int i = 0; i < cap_count; i++) {
        if (!(cap1[i].z >= 0 && cap1[i].z <= length)) {
            cap1_filtered[cap_filtered_count].x = cap1[i].x;
            cap1_filtered[cap_filtered_count].y = cap1[i].y;
            cap1_filtered[cap_filtered_count].z = cap1[i].z;
            cap_filtered_count++;
        }
    }

    cap_filtered_count = 0;
    for (int i = 0; i < cap_count; i++) {
        if (!(cap2[i].z >= 0 && cap2[i].z <= length)) {
            cap2_filtered[cap_filtered_count].x = cap2[i].x;
            cap2_filtered[cap_filtered_count].y = cap2[i].y;
            cap2_filtered[cap_filtered_count].z = cap2[i].z;
            cap_filtered_count++;
        }
    }

    //combine caps and cylinder
    int nCylinderPoints = nSlices * nCirclePoints;
    int nTotalPoints = nCylinderPoints + 2*cap_filtered_count;

    dfloat3* nodesTotal = (dfloat3*)malloc(nTotalPoints * sizeof(dfloat3));
    for (int i = 0; i < nCylinderPoints; i++) {
        nodesTotal[i].x = cylinder[i].x;
        nodesTotal[i].y = cylinder[i].y;
        nodesTotal[i].z = cylinder[i].z;
    }
    for (int i = 0; i < cap_filtered_count; i++) {
        nodesTotal[nCylinderPoints + i].x = cap1_filtered[i].x;
        nodesTotal[nCylinderPoints + i].y = cap1_filtered[i].y;
        nodesTotal[nCylinderPoints + i].z = cap1_filtered[i].z;
    }
    for (int i = 0; i < cap_filtered_count; i++) {
        nodesTotal[nCylinderPoints + cap_filtered_count + i].x = cap2_filtered[i].x;
        nodesTotal[nCylinderPoints + cap_filtered_count + i].y = cap2_filtered[i].y;
        nodesTotal[nCylinderPoints + cap_filtered_count + i].z = cap2_filtered[i].z;
    }

    //rotation
    //determine the quartetion which has to be used to rotate the vector (0,0,1) to vec
    //calculate dot product
    dfloat4 qf = compute_rotation_quart(dfloat3(0,0,1),vec);

    dfloat3 new_pos;
    for (int i = 0; i < nTotalPoints; i++) {
        new_pos = dfloat3(nodesTotal[i].x,nodesTotal[i].y,nodesTotal[i].z);
        
        new_pos = rotate_vector_by_quart_R(new_pos,qf);

        nodesTotal[i].x = new_pos.x + particleCenter->getSemiAxis1().x;
        nodesTotal[i].y = new_pos.y + particleCenter->getSemiAxis1().y;
        nodesTotal[i].z = new_pos.z + particleCenter->getSemiAxis1().z;

    }


    //DEFINITIONS
    
    dfloat volume = r*r*r*4*M_PI/3 + M_PI*r*r*length;
    dfloat sphereVol = r*r*r*4*M_PI/3;
    dfloat cylinderVol = M_PI*r*r*length;
    dfloat3 center = (particleCenter->getSemiAxis2() + particleCenter->getSemiAxis1())/2.0;

    pCenter->setRadius(r);
    pCenter->setVolume(sphereVol + cylinderVol);

    // Particle area
    pCenter->setS(4.0 * M_PI * r * r + 2*M_PI*r*length);
    // Particle density
    pCenter->setDensity(particleCenter->getDensity());

    // Particle center position
    pCenter->setPos(center);    
    pCenter->setPos_old(center);

    // Particle velocity
    pCenter->setVel(particleCenter->getVel());
    pCenter->setVel_old(particleCenter->getVel());
    
    // Particle rotation
    pCenter->setW(particleCenter->getW());
    pCenter->setW_avg(particleCenter->getW_avg());
    pCenter->setW_old(particleCenter->getW_old());

    pCenter->setQPosW(qf.w);
    pCenter->setQPosX(qf.x);
    pCenter->setQPosY(qf.y);
    pCenter->setQPosZ(qf.z);

    pCenter->setQPosOldW(pCenter->getQPosW());
    pCenter->setQPosOldX(pCenter->getQPosX());
    pCenter->setQPosOldY(pCenter->getQPosY());
    pCenter->setQPosOldZ(pCenter->getQPosZ());

    // Innertia momentum
    dfloat6 In;
    In.xx = pCenter->getDensity() * (cylinderVol*(r*r/2) + sphereVol * (2*r*r/5));
    In.yy = pCenter->getDensity() * (cylinderVol*(length*length/12 + r*r/4 ) + sphereVol * (2*r*r/5 + length*length/2 + 3*length*r/8));
    In.zz = pCenter->getDensity() * (cylinderVol*(length*length/12 + r*r/4 ) + sphereVol * (2*r*r/5 + length*length/2 + 3*length*r/8));

    In.xy = 0.0;
    In.xz = 0.0;
    In.yz = 0.0;

    dfloat4 q1 = compute_rotation_quart(dfloat3(1,0,0),vec);
    //rotate inertia 
    pCenter->setI(rotate_inertia_by_quart(q1,In));

    pCenter->setQPosW(qf.w);
    pCenter->setQPosX(qf.x);
    pCenter->setQPosY(qf.y);
    pCenter->setQPosZ(qf.z);

    pCenter->setQPosOldW(pCenter->getQPosW());
    pCenter->setQPosOldX(pCenter->getQPosX());
    pCenter->setQPosOldY(pCenter->getQPosY());
    pCenter->setQPosOldZ(pCenter->getQPosZ());
    
    pCenter->setFX(0.0);
    pCenter->setFY(0.0);
    pCenter->setFZ(0.0);

    pCenter->setFOldX(0.0);
    pCenter->setFOldY(0.0);
    pCenter->setFOldZ(0.0);

    pCenter->setMX(0.0);
    pCenter->setMY(0.0);
    pCenter->setMZ(0.0);

    pCenter->setMOldX(0.0);
    pCenter->setMOldY(0.0);
    pCenter->setMOldZ(0.0);


    // this->pCenter.movable = move;

    // this->pCenter.collision.shape = CAPSULE;
    // this->pCenter.collision.semiAxis = point1; 
    // this->pCenter.collision.semiAxis2 = point2; 
    // for(int i = 0; i <MAX_ACTIVE_COLLISIONS;i++){
    //     this->pCenter.collision.collisionPartnerIDs[i] = -1;
    //     this->pCenter.collision.tangentialDisplacements[i] = dfloat3(0,0,0);
    //     this->pCenter.collision.lastCollisionStep[i] = -1;
    // }

    this->numNodes = nTotalPoints;

    this->nodes = (IbmNodes*) malloc(sizeof(IbmNodes) * this->numNodes);

    //convert nodes info

    for (int nodeIndex = 0; nodeIndex < nTotalPoints; nodeIndex++) {
        
        this->nodes[nodeIndex].setPos(dfloat3(nodesTotal[nodeIndex].x,nodesTotal[nodeIndex].y,nodesTotal[nodeIndex].z));

        dfloat3 vel_nodes;
        vel_nodes.x = particleCenter->getVel().x + particleCenter->getW().y * this->nodes[nodeIndex].getPos().z - particleCenter->getW().z * this->nodes[nodeIndex].getPos().y;
        vel_nodes.y = particleCenter->getVel().y + particleCenter->getW().z * this->nodes[nodeIndex].getPos().x - particleCenter->getW().x * this->nodes[nodeIndex].getPos().z;
        vel_nodes.z = particleCenter->getVel().z + particleCenter->getW().x * this->nodes[nodeIndex].getPos().y - particleCenter->getW().y * this->nodes[nodeIndex].getPos().x;
       
        this->nodes[nodeIndex].setVel(vel_nodes);
        this->nodes[nodeIndex].setVelOld(vel_nodes);
       
        if(nodeIndex < nCylinderPoints)
            this->nodes[nodeIndex].setS((length*2*M_PI*r)/nCylinderPoints);
        else
            this->nodes[nodeIndex].setS((4*M_PI*r*r)/(2*cap_filtered_count));
    }

    //free
    free(z_cylinder);
    free(theta);
    free(cylinder);
    free(phi);
    free(cap1);
    free(cap2);
    free(cap1_filtered);
    free(cap2_filtered);
    free(nodesTotal);
 
     // Update old position value
     pCenter->setPos_old(pCenter->getPos());   

    for(int ii = 0;ii<nTotalPoints;ii++){
         IbmNodes* node_j = &(this->nodes[ii]);
    }

}

void Particle::makeEllipsoid(ParticleCenter *particleCenter)
{
    
    dfloat a, b, c;  // principal radius

    unsigned int i;

    a = particleCenter->getSemiAxis1().x;
    b = particleCenter->getSemiAxis1().y;
    c = particleCenter->getSemiAxis1().z;

    pCenter->setRadius(POW_FUNCTION(a*b*c,1.0/3.0));
    pCenter->setVolume(a*b*c*4*M_PI/3);

    // Particle density
    pCenter->setDensity(particleCenter->getDensity());//.density = density;

    // Particle center position
    pCenter->setPos(particleCenter->getPos());
    pCenter->setPos_old(particleCenter->getPos_old());

    // Particle velocity
    pCenter->setVel(particleCenter->getVel());
    pCenter->setVel_old(particleCenter->getVel_old());

    // Particle rotation
    pCenter->setW(particleCenter->getW());
    pCenter->setW_avg(particleCenter->getW_avg());
    pCenter->setW_old(particleCenter->getW_old());

    // Innertia momentum
    dfloat6 In;
    In.xx = 0.2 * pCenter->getVolume() * pCenter->getDensity() * (b*b + c*c);
    In.yy = 0.2 * pCenter->getVolume() * pCenter->getDensity() * (a*a + c*c);
    In.zz = 0.2 * pCenter->getVolume() * pCenter->getDensity() * (a*a + b*b);
    In.xy = 0.0;
    In.xz = 0.0;
    In.yz = 0.0;

    pCenter->setFX(0.0);
    pCenter->setFY(0.0);
    pCenter->setFZ(0.0);

    pCenter->setFOldX(0.0);
    pCenter->setFOldY(0.0);
    pCenter->setFOldZ(0.0);

    pCenter->setMX(0.0);
    pCenter->setMY(0.0);
    pCenter->setMZ(0.0);

    pCenter->setMOldX(0.0);
    pCenter->setMOldY(0.0);
    pCenter->setMOldZ(0.0);

    // Particle area
    dfloat p = 1.6075; //aproximation
    dfloat ab = POW_FUNCTION(a*b,p); 
    dfloat ac = POW_FUNCTION(a*c,p); 
    dfloat bc = POW_FUNCTION(b*c,p); 

    pCenter->setS(4*M_PI*POW_FUNCTION((ab + ac + bc)/3,1.0/p));

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dfloat scaling = myMin(myMin(a, b), c);
    
    a /= scaling;
    b /= scaling;
    c /= scaling;
    ab = POW_FUNCTION(a*b,p); 
    ac = POW_FUNCTION(a*c,p); 
    bc = POW_FUNCTION(b*c,p); 

    dfloat SS =  4*M_PI*POW_FUNCTION((ab + ac + bc)/3.0,1.0/p);
    int numberNodes = (int)(SS * POW_FUNCTION(scaling, 3.0) /(4*M_PI/MESH_SCALE));

    // Particle num nodes
    this->numNodes = numberNodes;

    //#############################################################################

    // Allocate memory for positions and forces
    dfloat *phi = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat *theta = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat* posx = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat* posy = (dfloat *)malloc(numberNodes* sizeof(dfloat));
    dfloat* posz = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    dfloat* fx = (dfloat *)malloc(numberNodes* sizeof(dfloat));
    dfloat* fy = (dfloat *)malloc(numberNodes* sizeof(dfloat));
    dfloat* fz = (dfloat *)malloc(numberNodes * sizeof(dfloat));
    
    // Initialize random positions of charges on the ellipsoid surface
    for (int i = 0; i < numberNodes; i++) {
        phi[i] = 2 * M_PI * ((dfloat)rand() / (dfloat)RAND_MAX);   // Angle in XY plane
        theta[i] = M_PI * ((dfloat)rand() / (dfloat)RAND_MAX);     // Angle from Z axis

        posx[i] = 1 * sin(theta[i]) * cos(phi[i]); // x coordinate
        posy[i] = 1 * sin(theta[i]) * sin(phi[i]); // y coordinate
        posz[i] = 1 * cos(theta[i]);               // z coordinate
    }

    // Constants
    dfloat base_k = 1.0; // Base Coulomb's constant (assuming unit charge)
    dfloat rij[3];
    dfloat r;
    dfloat F;
    dfloat unit_rij[3];
    dfloat force_scale_factor;

    for (int iter = 0; iter < 300; iter++) {
        // Initialize force accumulator
        for (int i = 0; i < numberNodes; i++) {
            fx[i] = 0.0;
            fy[i] = 0.0;
            fz[i] = 0.0;
        }

        // Compute pairwise forces and update positions
        for (int i = 0; i < numberNodes; i++) {
            for (int j = 0; j < numberNodes; j++) {
                if (i != j) { // not the same node
                    // Vector from node j to node i

                    rij[0] = posx[i] - posx[j];
                    rij[1] = posy[i] - posy[j];
                    rij[2] = posz[i] - posz[j];

                    // Distance between node i and node j
                    r = sqrt(rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2]);

                    // Coulomb's force magnitude
                    F = base_k / (r * r);

                    // Direction of force
                    unit_rij[0] = rij[0] / r;
                    unit_rij[1] = rij[1] / r;
                    unit_rij[2] = rij[2] / r;

                    // Accumulate force on nodes i
                    fx[i] += F * unit_rij[0];
                    fy[i] += F * unit_rij[1];
                    fz[i] += F * unit_rij[2];
                }
            }
        }
        // Update positions of nodes
        for (int i = 0; i < numberNodes; i++) {
            posx[i] += 10 * fx[i] / (numberNodes * numberNodes);
            posy[i] += 10 * fy[i] / (numberNodes * numberNodes);
            posz[i] += 10 * fz[i] / (numberNodes * numberNodes);
        }

        // Project updated positions back onto the ellipsoid surface
        for (int i = 0; i < numberNodes; i++) {
            // Calculate the current point's distance to the center along each axis
            force_scale_factor = sqrt(posx[i]*posx[i] +
                                posy[i]*posy[i] +
                                posz[i]*posz[i]);
            // Rescale to ensure it lies on the ellipsoid surface
            posx[i] /= force_scale_factor;
            posy[i] /= force_scale_factor;
            posz[i] /= force_scale_factor;
        }
    }
    //convert into elipsoid 
    for (int i = 0; i < numberNodes; i++) {
    posx[i] *= a*scaling;
    posy[i] *= b*scaling;
    posz[i] *= c*scaling;
    }
      

    this->nodes = (IbmNodes*) malloc(sizeof(IbmNodes) * this->numNodes);

    for (int nodeIndex = 0; nodeIndex < numberNodes; nodeIndex++) {
        this->nodes[nodeIndex].setPos(dfloat3(posx[nodeIndex],posy[nodeIndex],posz[nodeIndex]));

        dfloat3 vel_nodes;

        vel_nodes.x = particleCenter->getVel().x + particleCenter->getW().y * this->nodes[nodeIndex].getPos().z - particleCenter->getW().z * this->nodes[nodeIndex].getPos().y;
        vel_nodes.y = particleCenter->getVel().y + particleCenter->getW().z * this->nodes[nodeIndex].getPos().x - particleCenter->getW().x * this->nodes[nodeIndex].getPos().z;
        vel_nodes.z = particleCenter->getVel().z + particleCenter->getW().x * this->nodes[nodeIndex].getPos().y - particleCenter->getW().y * this->nodes[nodeIndex].getPos().x;

        this->nodes[nodeIndex].setVel(vel_nodes);
        this->nodes[nodeIndex].setVelOld(vel_nodes);

        // the area of sphere segment is divided by the number of node in the layer, so all nodes have the same area
        this->nodes[nodeIndex].setS(pCenter->getS() / numberNodes);
    }

    //%%%%%%%% ROTATION 
    //current state rotation quartenion (STANDARD ROTATION OF 90º IN THE X - AXIS)

    dfloat3 vec = dfloat3(particleCenter->getQ_pos().x,particleCenter->getQ_pos().y,particleCenter->getQ_pos().z);
    dfloat angleMag = particleCenter->getQ_pos().w;
    dfloat4 q2 = axis_angle_to_quart(vec,angleMag);

    //rotate inertia 
    pCenter->setI(rotate_inertia_by_quart(q2,In));

    dfloat3 new_pos;
    for (i = 0; i < numberNodes; i++) {

        new_pos = dfloat3(this->nodes[i].getPos().x,this->nodes[i].getPos().y,this->nodes[i].getPos().z);
        new_pos = rotate_vector_by_quart_R(new_pos,q2);

        this->nodes[i].setPos(dfloat3((new_pos.x + particleCenter->getPos().x),(new_pos.y + particleCenter->getPos().y),(new_pos.z + particleCenter->getPos().z)));
    }

    pCenter->setQPosW(q2.w);
    pCenter->setQPosX(q2.x);
    pCenter->setQPosY(q2.y);
    pCenter->setQPosZ(q2.z);

    pCenter->setQPosOldW(pCenter->getQPosW());
    pCenter->setQPosOldX(pCenter->getQPosX());
    pCenter->setQPosOldY(pCenter->getQPosY());
    pCenter->setQPosOldZ(pCenter->getQPosZ());

    pCenter->setSemiAxis1(particleCenter->getPos() + a*scaling*dfloat3(1,0,0));
    pCenter->setSemiAxis2(particleCenter->getPos() + b*scaling*dfloat3(0,1,0));
    pCenter->setSemiAxis3(particleCenter->getPos() + c*scaling*dfloat3(0,0,1));

    vec = vector_normalize(vec);
    if(angleMag != 0.0){
        const dfloat q0 = cos(0.5*angleMag);

        const dfloat qi = (vec.x/angleMag) * sin (0.5*angleMag);
        const dfloat qj = (vec.y/angleMag) * sin (0.5*angleMag);
        const dfloat qk = (vec.z/angleMag) * sin (0.5*angleMag);

        const dfloat tq0m1 = (q0*q0) - 0.5;

        pCenter->setSemiAxis1(particleCenter->getPos() + a*scaling*dfloat3(1,0,0));
        pCenter->setSemiAxis2(particleCenter->getPos() + b*scaling*dfloat3(0,1,0));
        pCenter->setSemiAxis3(particleCenter->getPos() + c*scaling*dfloat3(0,0,1));

        pCenter->setSemiAxis1(rotate_vector_by_quart_R(pCenter->getSemiAxis1() - particleCenter->getPos(),q2) + particleCenter->getPos());
        pCenter->setSemiAxis2(rotate_vector_by_quart_R(pCenter->getSemiAxis2() - particleCenter->getPos(),q2) + particleCenter->getPos());
        pCenter->setSemiAxis3(rotate_vector_by_quart_R(pCenter->getSemiAxis3() - particleCenter->getPos(),q2) + particleCenter->getPos());
    }


    // for(int i = 0; i <MAX_ACTIVE_COLLISIONS;i++){
    //     this->pCenter.collision.collisionPartnerIDs[i] = -1;
    //     this->pCenter.collision.tangentialDisplacements[i] = dfloat3(0,0,0);
    //     this->pCenter.collision.lastCollisionStep[i] = -1;
    // }

    // this->pCenter.collision.shape = ELLIPSOID;
    for(int ii = 0;ii<numberNodes;ii++){
         IbmNodes* node_j = &(this->nodes[ii]);
     }

    // Free allocated memory

    free(phi);
    free(theta);
    free(posx);
    free(posy);
    free(posz);
    free(fx);
    free(fy);
    free(fz);

     // Update old position value
    pCenter->setPos_old(pCenter->getPos());  
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}
#endif // !IBM

#endif //PARTICLE_MODEL