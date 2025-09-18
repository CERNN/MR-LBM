/*
for (int i = 0; i<NUM_PARTICLES;i++){
    ParticleCenter& center = centerStorage[i];

    this->pShape[i] = GRID;
    center.setDiameter(749056);
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);

}
*/

/*
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
*/

/**/

srand(12345);  // fixed seed for reproducibility


dfloat diameter = 20.0;
dfloat radius   = diameter / 2.0;

for (int i = 0; i < NUM_PARTICLES; i++) {
    ParticleCenter& center = centerStorage[i];
    dfloat3 pos;
    bool valid = false;

    while (!valid) {
        pos.x = (dfloat)((dfloat)rand() / RAND_MAX) * (NX - 1 - 2*radius) + radius;
        pos.y = (dfloat)((dfloat)rand() / RAND_MAX) * (NY - 1 - 2*radius) + radius;
        pos.z = (dfloat)((dfloat)rand() / RAND_MAX) * (NZ - 1 - 2*radius) + radius;
        
        valid = true;

        printf("Particle Pos %f %f %f \n",pos.x,pos.y,pos.z);

        for (int j = 0; j < i; j++) {
            dfloat3 otherPos = centerStorage[j].getPos();
            dfloat dx = pos.x - otherPos.x;
            dfloat dy = pos.y - otherPos.y;
            dfloat dz = pos.z - otherPos.z;
            dfloat dist = sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < diameter) {
                valid = false;
                break;
            }
        }
    }



    dfloat3 vel = {0.0, 0.0, 0.0};
    dfloat3 sa  = {10, 20, 30};

    this->pShape[i] = SPHERE;
    center.setPos(pos);
    center.setPos_old(pos);
    center.setVel(vel);
    center.setVel_old(vel);
    center.setDensity(1.166666);
    center.setDiameter(diameter);
    center.setSemiAxis1(sa);
    center.setMovable(false);
    center.setQ_pos(dfloat4(1,0,0,0));
        
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);
}