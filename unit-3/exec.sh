#!/bin/bash

# Array of CPU core counts
cpu_cores=(1 2 4 8)

# Output file
output_file="execution_times.txt"

# Clear the output file
> "$output_file"

# Loop through core counts
for num_cores in "${cpu_cores[@]}"; do
    # 
    output=$(mpirun -n "$num_cores" /opt/anaconda3/envs/py39/bin/python example_mpi12.py)

    # Extract execution time
    execution_time=$(echo "$output" | grep "Averaging result time:" | awk '{print $4}')

done

echo "Execution times saved to $output_file"
