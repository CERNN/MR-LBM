#ifndef __LBM_INITIALIZATION_H
#define __LBM_INITIALIZATION_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "moments.h"
#include "populations.h"

__global__
void gpuInitialization_mom(
    Moments mom
);

__global__
void gpuInitialization_pop(
    Moments mom,
    Populations pop
);

#endif // !__LBM_INITIALIZATION_H