#include "lbmInitialization.cuh"



__host__
void initializationRandomNumbers(
    float* randomNumbers, int seed)
{
    curandGenerator_t gen;

    // Create pseudo-random number generator
    checkCurandStatus(curandCreateGenerator(&gen,
        CURAND_RNG_PSEUDO_DEFAULT));
    
    // Set generator seed
    checkCurandStatus(curandSetPseudoRandomGeneratorSeed(gen,
        CURAND_SEED));
    
    // Generate NX*NY*NZ floats on device, using normal distribution
    // with mean=0 and std_dev=NORMAL_STD_DEV
    checkCurandStatus(curandGenerateNormal(gen, randomNumbers, NUMBER_LBM_NODES,
        0, CURAND_STD_DEV));

    checkCurandStatus(curandDestroyGenerator(gen));
}


__global__ void gpuInitialization_mom(
    dfloat *fMom, float* randomNumbers)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);
    //printf("threadIdx.x % d threadIdx.y % d threadIdx.z % d  bix %d biy %d biz %d --  x: %d y: %d z: %d idx %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, x, y, z, index);

    //first moments
    dfloat rho, ux, uy, uz;
    #ifdef NON_NEWTONIAN_FLUID
    dfloat omega;
    #endif

    //Taylor Green
	rho = RHO_0;
	ux = U_0_X;
	uy = U_0_Y;
    uz = U_0_Z;

    //if(y == NY-1 && (( x%(NX-1) != 0 ||z%(NZ-1) != 0)))
    //    ux = U_MAX;

    /*dfloat pert = 0.05;
    int l = idxScalarGlobal(x, y, z);
    int Nt = NUMBER_LBM_NODES;
    
    ux += (U_MAX)*pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
    uy += (U_MAX)*pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
    uz += (U_MAX)*pert*randomNumbers[l + z - Nt*((l + z) / Nt)];*/

    #ifdef NON_NEWTONIAN_FLUID
    omega = OMEGA;
    #endif

/*    
	rho = RHO_0 + (1.0/(16.0*cs2))*RHO_0*U_MAX*U_MAX*(cos((dfloat)2.0*(x) / L) + cos((dfloat)2.0*(y) / L))*(cos((dfloat)2.0*(z) / L) + 2.0);
	ux =   U_MAX * sin((dfloat)(x) / L) * cos((dfloat)(y) / L) * cos((dfloat)(z) / L);
	uy = - U_MAX * cos((dfloat)(x) / L) * sin((dfloat)(y) / L) * cos((dfloat)(z) / L);
    uz = 0.0;
*/    

    /*
    // Example of usage of random numbers for turbulence in parallel plates flow in z  
        dfloat y_visc = 6.59, ub_f = 15.6, uc_f = 18.2;
        // logaritimic velocity profile
        dfloat uz_log; 
        dfloat pos = (y < NY/2 ? y + 0.5 : NY - (y + 0.5));
        uz_log = -(uc_f*U_TAU)*(((pos-NY/2)/del)*((pos-NY/2)/del)) + (uc_f*U_TAU);
        
        uz = uz_log;
        ux = 0.0;
        uy = 0.0;
        rho = RHO_0;


        // perturbation
        dfloat pert = 0.1;
        int l = idxScalarGlobal(x, y, z);
        int Nt = NUMBER_LBM_NODES;
        uz += (ub_f*U_TAU)*pert*randomNumbers[l + z - Nt*((l + z) / Nt)];
        ux += (ub_f*U_TAU)*pert*randomNumbers[l + x - Nt*((l + x) / Nt)];
        uy += (ub_f*U_TAU)*pert*randomNumbers[l + y - Nt*((l + y) / Nt)];
    */   

    
    // zeroth moment
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = rho-RHO_0;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = ux;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = uz;

    //second moments
    //define equilibrium populations
    dfloat pop[Q];
    for (int i = 0; i < Q; i++)
    {
        pop[i] = gpu_f_eq(w[i] * RHO_0,
                          3 * (ux * cx[i] + uy * cy[i] + uz * cz[i]),
                          1 - 1.5 * (ux * ux + uy * uy + uz * uz));
    }
    
    dfloat invRho = 1.0/rho;
    dfloat pixx =  (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16]) * invRho - cs2;
    dfloat pixy = ((pop[7] + pop[ 8]) - (pop[13] + pop[14])) * invRho;
    dfloat pixz = ((pop[9] + pop[10]) - (pop[15] + pop[16])) * invRho;
    dfloat piyy =  (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18]) * invRho - cs2;
    dfloat piyz = ((pop[11]+pop[12])-(pop[17]+pop[18])) * invRho;
    dfloat pizz =  (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18]) * invRho - cs2;

    //pixx = pixx + OMEGA * (RHO_0 * ux * ux -  pixx)  + TT_OMEGA * (FX * ux + FX * ux);
    //pixy = pixy + OMEGA * (RHO_0 * ux * uy -  pixy)  + TT_OMEGA * (FX * uy + FY * ux);
    //pixz = pixz + OMEGA * (RHO_0 * ux * uz -  pixz)  + TT_OMEGA * (FX * uz + FZ * ux);
    //piyy = piyy + OMEGA * (RHO_0 * uy * uy -  piyy)  + TT_OMEGA * (FY * uy + FY * uy);
    //piyz = piyz + OMEGA * (RHO_0 * uy * uz -  piyz)  + TT_OMEGA * (FY * uz + FZ * uy);
    //pizz = pizz + OMEGA * (RHO_0 * uz * uz -  pizz)  + TT_OMEGA * (FZ * uz + FZ * uz);

    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pixx; //= RHO_0*ux*ux+RHO_0*cs2;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pixy; //= RHO_0*ux*uy;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pixz; //= RHO_0*ux*uz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = piyy; //= RHO_0*uy*uy+RHO_0*cs2;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = piyz; //= RHO_0*uy*uz;
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = pizz; //= RHO_0*uz*uz+RHO_0*cs2;

    #ifdef NON_NEWTONIAN_FLUID
    fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_OMEGA_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)] = omega;
    #endif   
}

__global__ void gpuInitialization_pop(
    dfloat *fMom,
    dfloat *fGhostX_0, dfloat *fGhostX_1,
    dfloat *fGhostY_0, dfloat *fGhostY_1,
    dfloat *fGhostZ_0, dfloat *fGhostZ_1)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);
    // zeroth moment

    dfloat rhoVar = RHO_0 + fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_RHO_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uxVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uyVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat uzVar = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_UZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixx = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXX_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pixz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MXZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyy = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYY_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat piyz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MYZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];
    dfloat pizz = fMom[idxMom(threadIdx.x, threadIdx.y, threadIdx.z, M_MZZ_INDEX, blockIdx.x, blockIdx.y, blockIdx.z)];

    dfloat pop[Q];
    #pragma unroll //equation 6
    for (int i = 0; i < Q; i++)
    {
        pop[i] = rhoVar * w[i] * (1 
        + as2 * (uxVar * cx[i] + uyVar * cy[i] + uzVar * cz[i]) 
        + 0.5 * as2 * as2 * (
            pixx * (cx[i] * cx[i] - cs2) + 
            2.0*pixy * (cx[i] * cy[i]) + 
            2.0*pixz * (cx[i] * cz[i]) + 
            piyy * (cy[i] * cy[i] - cs2) + 
            2.0*piyz * (cy[i] * cz[i]) + 
            pizz * (cz[i] * cz[i] - cs2))
        );
    }

    
    //thread xyz
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    
    //block xyz
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    if (threadIdx.x == 0) { //w
        fGhostX_0[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 2]; 
        fGhostX_0[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 8];
        fGhostX_0[idxPopX(ty, tz, 2, bx, by, bz)] = pop[10];
        fGhostX_0[idxPopX(ty, tz, 3, bx, by, bz)] = pop[14];
        fGhostX_0[idxPopX(ty, tz, 4, bx, by, bz)] = pop[16];
        #ifdef D3Q27                                                                                                           
        fGhostX_0[idxPopX(ty, tz, 5, bx, by, bz)] = pop[20];
        fGhostX_0[idxPopX(ty, tz, 6, bx, by, bz)] = pop[22];
        fGhostX_0[idxPopX(ty, tz, 7, bx, by, bz)] = pop[24];
        fGhostX_0[idxPopX(ty, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.x == (BLOCK_NX - 1)){                                                                                                                                                                               
        fGhostX_1[idxPopX(ty, tz, 0, bx, by, bz)] = pop[ 1];
        fGhostX_1[idxPopX(ty, tz, 1, bx, by, bz)] = pop[ 7];
        fGhostX_1[idxPopX(ty, tz, 2, bx, by, bz)] = pop[ 9];
        fGhostX_1[idxPopX(ty, tz, 3, bx, by, bz)] = pop[13];
        fGhostX_1[idxPopX(ty, tz, 4, bx, by, bz)] = pop[15];
        #ifdef D3Q27                                                                                                           
        fGhostX_1[idxPopX(ty, tz, 5, bx, by, bz)] = pop[19];
        fGhostX_1[idxPopX(ty, tz, 6, bx, by, bz)] = pop[21];
        fGhostX_1[idxPopX(ty, tz, 7, bx, by, bz)] = pop[23];
        fGhostX_1[idxPopX(ty, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27       
    }

    if (threadIdx.y == 0)  { //s                                                                                                                                                                                        
        fGhostY_0[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 4];
        fGhostY_0[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 8];
        fGhostY_0[idxPopY(tx, tz, 2, bx, by, bz)] = pop[12];
        fGhostY_0[idxPopY(tx, tz, 3, bx, by, bz)] = pop[13];
        fGhostY_0[idxPopY(tx, tz, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        fGhostY_0[idxPopY(tx, tz, 5, bx, by, bz)] = pop[20];
        fGhostY_0[idxPopY(tx, tz, 6, bx, by, bz)] = pop[22];
        fGhostY_0[idxPopY(tx, tz, 7, bx, by, bz)] = pop[23];
        fGhostY_0[idxPopY(tx, tz, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.y == (BLOCK_NY - 1)){                                                                                                                                                                        
        fGhostY_1[idxPopY(tx, tz, 0, bx, by, bz)] = pop[ 3];
        fGhostY_1[idxPopY(tx, tz, 1, bx, by, bz)] = pop[ 7];
        fGhostY_1[idxPopY(tx, tz, 2, bx, by, bz)] = pop[11];
        fGhostY_1[idxPopY(tx, tz, 3, bx, by, bz)] = pop[14];
        fGhostY_1[idxPopY(tx, tz, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        fGhostY_1[idxPopY(tx, tz, 5, bx, by, bz)] = pop[19];
        fGhostY_1[idxPopY(tx, tz, 6, bx, by, bz)] = pop[21];
        fGhostY_1[idxPopY(tx, tz, 7, bx, by, bz)] = pop[24];
        fGhostY_1[idxPopY(tx, tz, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                           
    }
    
    if (threadIdx.z == 0){ //b                                                                                                                                                                                     
        fGhostZ_0[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 6];
        fGhostZ_0[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[10];
        fGhostZ_0[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[12];
        fGhostZ_0[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[15];
        fGhostZ_0[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[17];
        #ifdef D3Q27                                                                                                           
        fGhostZ_0[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[20];
        fGhostZ_0[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[21];
        fGhostZ_0[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[24];
        fGhostZ_0[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[26];
        #endif //D3Q27                                                                                                           
    }else if (threadIdx.z == (BLOCK_NZ - 1)){                                                                                                               
        fGhostZ_1[idxPopZ(tx, ty, 0, bx, by, bz)] = pop[ 5];
        fGhostZ_1[idxPopZ(tx, ty, 1, bx, by, bz)] = pop[ 9];
        fGhostZ_1[idxPopZ(tx, ty, 2, bx, by, bz)] = pop[11];
        fGhostZ_1[idxPopZ(tx, ty, 3, bx, by, bz)] = pop[16];
        fGhostZ_1[idxPopZ(tx, ty, 4, bx, by, bz)] = pop[18];
        #ifdef D3Q27                                                                                                           
        fGhostZ_1[idxPopZ(tx, ty, 5, bx, by, bz)] = pop[19];
        fGhostZ_1[idxPopZ(tx, ty, 6, bx, by, bz)] = pop[22];
        fGhostZ_1[idxPopZ(tx, ty, 7, bx, by, bz)] = pop[23];
        fGhostZ_1[idxPopZ(tx, ty, 8, bx, by, bz)] = pop[25];
        #endif //D3Q27                                                                                                                                                                                                                    
    }
}


__global__ void gpuInitialization_nodeType(
    unsigned int *dNodeType)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;
    
    unsigned int nodeType;

    #include BC_INIT_PATH

    dNodeType[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = nodeType;
}


__global__ void gpuInitialization_force(
    dfloat *d_BC_Fx, dfloat* d_BC_Fy, dfloat* d_BC_Fz)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    size_t index = idxScalarGlobal(x, y, z);

    d_BC_Fx[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0;
    d_BC_Fy[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0;
    d_BC_Fz[idxScalarBlock(threadIdx.x, threadIdx.y, threadIdx.z,blockIdx.x, blockIdx.y, blockIdx.z)] = 0.0; 
}


__host__ void hostInitialization_nodeType(
    unsigned int *hNodeType)
{
    int x,y,z;
    unsigned int nodeType;

    for (x = 0; x<NX;x++){
        for (y = 0; y<NY;y++){
            for (z = 0; z<NZ_TOTAL;z++){
                #include BC_INIT_PATH
                hNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] = (unsigned int)nodeType;
            }
        }
    }

    printf("bulk boundary condition done\n");
}


void read_xyz_file(
    const std::string& filename,
    unsigned int* dNodeType
) {
    std::ifstream csv_file(filename);
    if (!csv_file)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    int x, y, z;
    int value;
    size_t index, index_n;

    int xi, yi, zi;

    std::string line;
    while (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string field;

        std::getline(ss, field, ',');
        x = std::stol(field);

        std::getline(ss, field, ',');
        y = std::stol(field);

        std::getline(ss, field, ',');
        z = std::stol(field);


        index = idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ);
        //printf("x %d y %d z %d \n",x,y,z); fflush(stdout);
        dNodeType[idxScalarBlock(x % BLOCK_NX, y % BLOCK_NY, z % BLOCK_NZ, x / BLOCK_NX, y / BLOCK_NY, z / BLOCK_NZ)] = SOLID_NODE;


        //set neighborings to be BC
        for (int xn = -1; xn < 2; xn++) {
            for (int yn = -1; yn < 2; yn++) {
                for (int zn = -1; zn < 2; zn++) {

                    xi = (x + xn + NX) % NX;
                    yi = (y + yn + NY) % NY;
                    zi = (z + zn + NZ) % NZ;


                    index_n = idxScalarBlock(xi% BLOCK_NX, yi % BLOCK_NY, zi % BLOCK_NZ, xi / BLOCK_NX, yi / BLOCK_NY, zi / BLOCK_NZ);

                    if ((index_n == index) || dNodeType[index_n] == 255) // check if is the center of the cuboid or if is already a solid node
                        continue;
                    else //set flag to max int 
                        dNodeType[index_n] = MISSING_DEFINITION;
                }
            }
        }
    }
    csv_file.close();
    printf("voxels imported \n");
}

/*
void define_voxel_bc(
    unsigned int *dNodeType
){
    for(int x= 0;x<NX;x++){
        for(int y =0; y<NY;y++){
            for(int z =0; z<NZ_TOTAL;z++){
                unsigned int index = idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ);
                if(dNodeType[index] == MISSING_DEFINITION){
                    dNodeType[index] = bc_id(dNodeType,x,y,z);
                }
            }
        }
    }
}
*/

__global__ 
void define_voxel_bc(
    unsigned int *dNodeType
){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= NX || y >= NY || z >= NZ)
        return;

    unsigned int index = idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ);
    if(dNodeType[index] == MISSING_DEFINITION){
        dNodeType[index] = bc_id(dNodeType,x,y,z);
    }
}



/*
Note: Due to the way the BC are set up, it possible when setting a solid node to also set the bit flags of neighboring nodes
However if attempt to perform in device, need to pay attention of two solid nodes setting the same flag at same time 
*/
__host__ __device__
unsigned int bc_id(unsigned int *dNodeType, int x, int y, int z){

    unsigned int bc_d = BULK;

    int xp1 = (x+1+NX)%NX;
    int xm1 = (x-1+NX)%NX;
    int yp1 = (y+1+NY)%NY;
    int ym1 = (y-1+NY)%NY;
    int zp1 = (z+1+NZ)%NZ;
    int zm1 = (z-1+NZ)%NZ;

    // 1
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 3);
        bc_d |= (1 << 5);
        bc_d |= (1 << 7);
    }
     // 2
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 2);
        bc_d |= (1 << 4);
        bc_d |= (1 << 6);
    }
    // 3
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 4
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
    }
    // 5
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 6
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
    }
    // 7
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 3);
        bc_d |= (1 << 7);
    }
    // 8
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 4);
    }
    // 9
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 5);
        bc_d |= (1 << 7);
    }
    // 10
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 2);
    }
    // 11
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 6);
        bc_d |= (1 << 7);
    }
    // 12
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
        bc_d |= (1 << 1);
    }
    // 13
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, z%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 5);
    }
    // 14
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, z%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, z/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 6);
    }
    // 15
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, y%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, y/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);
        bc_d |= (1 << 3);
    }
    // 16
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, y%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, y/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 6);
    }
    // 17
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, x/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
        bc_d |= (1 << 3);
    }
    // 18
    if(dNodeType[idxScalarBlock(x%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, x/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
        bc_d |= (1 << 5);
    }
    // 19
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 7);
    }
    // 20
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 0);
    }
    // 21
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, yp1%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, yp1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 3);
    }
    // 22
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 4);
    }
    // 23
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 5);
    }
    // 24
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zp1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 2);
    }
    // 25
    if(dNodeType[idxScalarBlock(xm1%BLOCK_NX, yp1%BLOCK_NY, zp1%BLOCK_NZ, xm1/BLOCK_NX, yp1/BLOCK_NY, zp1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 6);
    }
    // 26
    if(dNodeType[idxScalarBlock(xp1%BLOCK_NX, ym1%BLOCK_NY, zm1%BLOCK_NZ, xp1/BLOCK_NX, ym1/BLOCK_NY, zm1/BLOCK_NZ)] == 255){
        bc_d |= (1 << 1);   
    }

    return bc_d;
}