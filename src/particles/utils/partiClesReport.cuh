/*
*   @file ibmReport.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Functions for reporting informations about IBM particles
*   @version 0.3.0
*   @date 13/10/2020
*/


#ifndef __PARTICLES_REPORT_H
#define __PARTICLES_REPORT_H

#include <fstream>
#include <string>
#include <sstream>
//#include "../lbmReport.h"
#include "../class/Particle.cuh"
#include "../../saveData.cuh"

#ifdef PARTICLE_MODEL

/**
*   @brief Save particles informations
*   
*   @param particles: particles array
*   @param step: current time step
*/
void saveParticlesInfo(ParticlesSoA *particles, unsigned int step);

/**
*   @brief Print particles information
*   
*   @param particles: particles array
*   @param step: current time step
*/
void printParticlesInfo(ParticlesSoA particles, unsigned int step);

#endif
#endif //PARTICLE_MODEL
