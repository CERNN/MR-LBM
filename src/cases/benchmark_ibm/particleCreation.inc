for (int i = 0; i<NUM_PARTICLES;i++){
    ParticleCenter& center = centerStorage[i];

    this->pShape[i] = RANDOM;
    center.setDiameter(80000);
    particles[i].setMethod(IBM);
    particles[i].setShape(&pShape[i]);
    particles[i].setPCenter(&center);
    particles[i].setCollideParticle(false);
    particles[i].setCollideWall(true);

}
