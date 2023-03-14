CC=86

if [[ "$1" = "D3Q19" || "$1" = "D3Q27" ]]
then
    nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v -O3 --restrict \
        *.cu \
        -lcudadevrt -lcurand -o ./../bin/$2sim_$1_sm${CC}
else
    echo "Input error, example of usage is"
    echo "sh compile.sh D3Q19 011"
    echo "sh compile.sh D3Q27 202"
fi
