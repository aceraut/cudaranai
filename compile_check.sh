#!/bin/bash

# Define the source and object directories
SRC_DIR="./core"
OBJ_DIR="./obj"

# Create the object directory if it does not exist
mkdir -p "$OBJ_DIR"

# Compile each .cu file in the source directory
for src_file in "$SRC_DIR"/*.cu; do
    if [[ -f "$src_file" ]]; then
        # Extract the base filename without extension
        base_filename=$(basename "$src_file" .cu)
        
        # Define the object file path
        obj_file="$OBJ_DIR/$base_filename.o"
        
        # Compile the source file to the object file using nvcc with the --expt-extended-lambda flag
        nvcc --expt-extended-lambda -g -G -c "$src_file" -o "$obj_file"
        
        # Check if the compilation was successful
        if [[ $? -eq 0 ]]; then
            echo "Compiled $src_file to $obj_file"
        else
            echo "Failed to compile $src_file"
        fi
    fi
done
