/**
*   @file particlesReport.cuh
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
*   @author report about particles
*   @brief report data about particles
*   @version 0.4.0
*   @date 01/09/2025
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
