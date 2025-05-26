# To compile in linux:
# dos2unix compile_linux.sh
# chmod +x compile_linux.sh D3Q19 000
# ../bin/000sim_D3Q19_sm80
# edit where necessary

# CC=86

#!/bin/bash

# Set CC if it's not already defined
if [ -z "$CC" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -z "$CC" ]; then
        echo "Error: Unable to determine compute capability."
        exit 1
    fi
fi

# Check if the argument is D3Q19 or D3Q27
if [[ "$1" == "D3Q19" || "$1" == "D3Q27" ]]; then
    nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -O3 --restrict  -DSM_${CC} \
        -DTARGET_LINUX \
        *.cu \
        -diag-suppress=39 \
        -diag-suppress=179 \
        -lcudadevrt -lcurand -o ./../bin/"$2"sim_"$1"_sm${CC}
else
    echo "Input error, example of usage is:"
    echo "sh compile.sh D3Q19 011"
    echo "sh compile.sh D3Q27 202"
fi


#--ptxas-options=-v
# 39,179 suppress division by false in the mods

        # -diag-suppress 550 \
        # -diag-suppress 549 \
        # -diag-suppress 177 \
 #               -diag-suppress 39 \
#        -diag-suppress 179 \
        # -lineinfo \ #usefull for nsight compute debug