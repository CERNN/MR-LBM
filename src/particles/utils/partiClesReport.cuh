/**
 *  @file particlesReport.cuh
 *  Contributors history:
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @brief report data about particles
 *  @version 0.4.0
 *  @date 01/09/2025
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
 *  @brief Convert a dfloat3 structure to a string representation with specified separator.
 *  @param val The dfloat3 structure to convert.
 *  @param sep The separator to use between values.
 *  @return A string representation of the dfloat3 structure.
 */
std::string getStrDfloat3(dfloat3 val, std::string sep);

/**
 *  @brief Convert a dfloat4 structure to a string representation with specified separator.
 *  @param val The dfloat4 structure to convert.
 *  @param sep The separator to use between values.
 *  @return A string representation of the dfloat4 structure.
 */
std::string getStrDfloat4(dfloat4 val, std::string sep);

/**
 *  @brief Convert a dfloat6 structure to a string representation with specified separator.
 *  @param val The dfloat6 structure to convert.
 *  @param sep The separator to use between values.
 *  @return A string representation of the dfloat6 structure.
 */
std::string getStrDfloat6(dfloat6 val, std::string sep);

/**
 *  @brief Save particles informations
*   
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param step: Current time step
*/
void saveParticlesInfo(ParticlesSoA *particles, unsigned int step);

/**
 *  @brief Print particles information 
 *  @param particles: Pointer to the ParticlesSoA structure containing particle data.
 *  @param step: Current time step
 */
void printParticlesInfo(ParticlesSoA particles, unsigned int step);

#endif //PARTICLE_MODEL
#endif //!__PARTICLES_REPORT_H
