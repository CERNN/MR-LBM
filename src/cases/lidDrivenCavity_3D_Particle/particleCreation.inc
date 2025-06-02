// file where you describe your particles

std::random_device rand_dev;
std::minstd_rand generator(rand_dev());
std::uniform_int_distribution<int>  distr(0, RAND_MAX);

ParticleCenter pc_i;


dfloat x_limit_B = 0 + 5 / 2.0;
dfloat x_limit_E = NX - 5 / 2.0;
dfloat y_limit_B = 0 + 5 / 2.0;
dfloat y_limit_E = NY - 5 / 2.0;
dfloat z_limit_B = 0 + 5 / 2.0;
dfloat z_limit_E = NZ - 5 / 2.0;

dfloat x_center;
dfloat y_center;
dfloat z_center;


for (int i = 0; i<NUM_PARTICLES;i++){

    x_center = x_limit_B + (x_limit_E - x_limit_B) * distr(generator) / RAND_MAX;
    y_center = y_limit_B + (y_limit_E - y_limit_B) * distr(generator) / RAND_MAX;
    z_center = z_limit_B + (z_limit_E - z_limit_B) * distr(generator) / RAND_MAX;

    pc_i.setPos(dfloat3(x_center,y_center,z_center));
    pc_i.setPos_old(dfloat3(x_center,y_center,z_center));
    pc_i.setVel(dfloat3(0,0,0));
    pc_i.setVel_old(dfloat3(0,0,0));


    particles[i].setMethod(TRACER);
    particles[i].setPCenter(pc_i);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);



}