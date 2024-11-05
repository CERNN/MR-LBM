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
    nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v -O3 --restrict \
        *.cu \
        -lcudadevrt -lcurand -o ./../bin/$2sim_$1_sm${CC}
else
    echo "Input error, example of usage is:"
    echo "sh compile.sh D3Q19 011"
    echo "sh compile.sh D3Q27 202"
fi