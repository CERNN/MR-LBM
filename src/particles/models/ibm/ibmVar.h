/*
*   @file ibmVar.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Configurations for the immersed boundary method
*   @version 0.3.0
*   @date 26/08/2020
*/

#ifndef __IBM_VAR_H
#define __IBM_VAR_H

#include "../../../var.h"
#include <stdio.h>
#include <math.h>


// Change to location of nodes http://dx.doi.org/10.1016/j.jcp.2012.02.026
#define BREUGEM_PARAMETER (0.0)
#define IBM_DISC_RELAXATION (0.5) // 0 = Leap-frog, 0.5 = Crank-Nicolson, 1 = Euler //NOT BEING USED CURRENTLY
// Mesh scale for IBM, minimum distance between nodes (lower, more nodes in particle)
#define MESH_SCALE 1.0
// Assumed boundary thickness for IBM
#define IBM_THICKNESS (1)
// Number of iterations of Coulomb algorithm to optimize the nodes positions
#define MESH_COULOMB 0
// Number of IBM inner iterations
#define IBM_MAX_ITERATION 1
//#define IBM_DEBUG
#define MACR_BORDER_NODES (2+(int)((IBM_EULER_UPDATE_DIST+IBM_PARTICLE_SHELL_THICKNESS)+0.99999999))




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

#endif // !__IBM_VAR_H
