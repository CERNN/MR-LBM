# To compile in linux:
# dos2unix compile_linux.sh
# chmod +x compile_linux.sh D3Q19 000
# ../bin/000sim_D3Q19_sm80
# edit where necessary

# CC=86


nvcc -gencode arch=compute_80,code=sm_80 -rdc=true -O3 --restrict  -DSM_${CC} \
    -DTARGET_LINUX \
    *.cu \
    -diag-suppress=39 \
    -diag-suppress=179 \
    -lcudadevrt -lcurand -o ./../bin/"$1"sim_D3Q19_sm80
