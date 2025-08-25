#!/bin/bash

# Check if model is provided
if [ -z "$1" ]; then
    echo "Error: Model not specified."
    echo "Usage: sh compile.sh <model> <output_prefix>"
    echo "Example: sh compile.sh D3Q19 011"
    echo "Example: sh compile.sh D3Q27 202"
    exit 1
fi

# Check if model is D3Q19 or D3Q27
if [[ "$1" != "D3Q19" && "$1" != "D3Q27" ]]; then
    echo "Input error, model must be D3Q19 or D3Q27"
    echo "Usage: sh compile.sh <model> <output_prefix>"
    echo "Example: sh compile.sh D3Q19 011"
    echo "Example: sh compile.sh D3Q27 202"
    exit 1
fi

# Check if output prefix is provided
if [ -z "$2" ]; then
    echo "Error: Output prefix not specified."
    echo "Usage: sh compile.sh <model> <output_prefix>"
    echo "Example: sh compile.sh D3Q19 011"
    echo "Example: sh compile.sh D3Q27 202"
    exit 1
fi

# Determine compute capability if not set
if [ -z "$CC" ]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    if [ -z "$CC" ]; then
        echo "Error: Unable to determine compute capability."
        exit 1
    fi
fi

# Detect host platform
HOST_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
if [[ "$HOST_OS" == *"linux"* ]]; then
    PLATFORM="linux"
    OUTPUT_EXT=""
    TARGET_FLAG="-DTARGET_LINUX"
elif [[ "$HOST_OS" == *"cygwin"* || "$HOST_OS" == *"mingw"* || "$OSTYPE" == *"msys"* ]]; then
    PLATFORM="windows"
    OUTPUT_EXT=".exe"
    TARGET_FLAG=""
else
    echo "Error: Unable to detect host platform. This script supports Linux or Windows (via Cygwin/MSYS2/WSL)."
    exit 1
fi

# Compile with nvcc
nvcc -gencode arch=compute_${CC},code=sm_${CC} -rdc=true -O3 --restrict -DSM_${CC} \
    ${TARGET_FLAG} \
    *.cu \
    -diag-suppress=39 \
    -diag-suppress=179 \
    -lcudadevrt -lcurand -o ./../bin/${2}sim_${1}_sm${CC}${OUTPUT_EXT}

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful for $PLATFORM: ./../bin/${2}sim_${1}_sm${CC}${OUTPUT_EXT}"
else
    echo "Compilation failed."
    exit 1
fi