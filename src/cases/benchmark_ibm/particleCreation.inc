// for (int i = 0; i<NUM_PARTICLES;i++){
//     ParticleCenter& center = centerStorage[i];

//     this->pShape[i] = RANDOM;
//     center.setDiameter(80000);
//     particles[i].setMethod(IBM);
//     particles[i].setShape(&pShape[i]);
//     particles[i].setPCenter(&center);
//     particles[i].setCollideParticle(false);
//     particles[i].setCollideWall(true);

// }

dfloat3 pos[NUM_PARTICLES];

pos[0] = {(NX-1)/2.0,65,(NZ-1)/2.0};
pos[1] = {(NX-1)/2.0,165,(NZ-1)/2.0};



dfloat3 vel = {0.0, 0.0, 0.0};
dfloat3 w = {0.0, 0.0, 0.0};


for (int i = 0; i<NUM_PARTICLES;i++){
    ParticleCenter& center = centerStorage[i];

    dfloat3 sa = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setPos(pos[i]);
    center.setPos_old(pos[i]);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(30);
    center.setSemiAxis1(sa);
    center.setMovable(true);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);

}