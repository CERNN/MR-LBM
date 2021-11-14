#ifndef __MLBM_H
#define __MLBM_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "moments.h"

__global__
void gpuMomCollisionStream(
    Moments mom
);


#endif __MLBM_H