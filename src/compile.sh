# CC=86

# Set CC if it's not already defined
if [ -z "$CC" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -z "$CC" ]; then
        echo "Error: Unable to determine compute capability."
        exit 1
    fi
fi

if [[ "$1" = "D3Q19" || "$1" = "D3Q27" ]]; then
    nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -O3 --restrict -DSM_${CC}  \
        *.cu \
        -diag-suppress 39 \
        -diag-suppress 179 \
        -lcudadevrt -lcurand -o ./../bin/$2sim_$1_sm${CC}
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
        # -lineinfo \ #usefull for nsight compute debug