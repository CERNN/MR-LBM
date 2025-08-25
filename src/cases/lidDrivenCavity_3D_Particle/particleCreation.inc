/*
centerStorage.resize(NUM_PARTICLES);
pShape = new ParticleShape[NUM_PARTICLES];

std::random_device rand_dev;
std::minstd_rand generator(rand_dev());
std::uniform_int_distribution<int>  distr(0, RAND_MAX);

//dfloat x_limit_B = 0 + 5 / 2.0;
//dfloat x_limit_E = NX - 5 / 2.0;
//dfloat y_limit_B = 0 + 5 / 2.0;
//dfloat y_limit_E = NY - 5 / 2.0;
//dfloat z_limit_B = 0 + 5 / 2.0;
//dfloat z_limit_E = NZ - 5 / 2.0;

dfloat x_center;
dfloat y_center;
dfloat z_center;

for (int i = 0; i<NUM_PARTICLES;i++){
    ParticleCenter& center = centerStorage[i];
    

    x_center = 64; // x_limit_B + (x_limit_E - x_limit_B) * distr(generator) / RAND_MAX;
    y_center = 120; //y_limit_B + (y_limit_E - y_limit_B) * distr(generator) / RAND_MAX;
    z_center = 64; //z_limit_B + (z_limit_E - z_limit_B) * distr(generator) / RAND_MAX;

    dfloat3 pos = {x_center, y_center, z_center};
    dfloat3 vel = {0.0, 0.0, 0.0};

    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    
    particles[i].setMethod(TRACER);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);

}
*/