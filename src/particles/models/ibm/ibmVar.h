/**
 *  @file ibmVar.h
 *  @author Waine Jr. (waine@alunos.utfpr.edu.br)
 *  @author Marco Aurelio Ferrari (e.marcoferrari@utfpr.edu.br)
 *  @author Ricardo de Souza
 *  @brief variables for IBM
 *  @version 0.4.0
 *  @date 01/09/2025
 */


#ifndef __IBM_VAR_H
#define __IBM_VAR_H

#include "../../../var.h"
#include <stdio.h>
#include <math.h>

#ifdef PARTICLE_MODEL

// Number of IBM inner iterations
#define IBM_MAX_ITERATION 1 //TODO: cant handle more than one iteration
//#define IBM_DEBUG


/* ------------------------ THREADS AND GRIDS FOR IBM ----------------------- */
// Threads for IBM particles
constexpr unsigned int THREADS_PARTICLES_IBM = NUM_PARTICLES > 64 ? 64 : NUM_PARTICLES;
// Grid for IBM particles
constexpr unsigned int GRID_PARTICLES_IBM = 
    (NUM_PARTICLES % THREADS_PARTICLES_IBM ? 
        (NUM_PARTICLES / THREADS_PARTICLES_IBM + 1)
        : (NUM_PARTICLES / THREADS_PARTICLES_IBM));

// For IBM particles collision, the total of threads must be 
// totalThreads = NUM_PARTICLES*(NUM_PARTICLES+1)/2
constexpr unsigned int TOTAL_PCOLLISION_IBM_THREADS = (NUM_PARTICLES*(NUM_PARTICLES+1))/2;
// Threads for IBM particles collision 
constexpr unsigned int THREADS_PCOLLISION_IBM = (TOTAL_PCOLLISION_IBM_THREADS > 64) ? 
    64 : TOTAL_PCOLLISION_IBM_THREADS;
// Grid for IBM particles collision
constexpr unsigned int GRID_PCOLLISION_IBM = 
    (TOTAL_PCOLLISION_IBM_THREADS % THREADS_PCOLLISION_IBM ? 
        (TOTAL_PCOLLISION_IBM_THREADS / THREADS_PCOLLISION_IBM + 1)
        : (TOTAL_PCOLLISION_IBM_THREADS / THREADS_PCOLLISION_IBM));
/* -------------------------------------------------------------------------- */

#endif //PARTICLE_MODEL
#endif // !__IBM_VAR_H


