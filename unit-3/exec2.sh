#!/bin/bash

# Array of CPU core counts
cpu_cores=(1 2 4 8)

# Output file
output_file="execution_times.txt"

# Clear the output file
> "$output_file"

# Loop through core counts
for num_cores in "${cpu_cores[@]}"; do
    # Run the MPI script and capture output
    output=$(mpirun -n "$num_cores" python example_mpi12.py)

    # Extract execution time
    execution_time=$(echo "$output" | grep "Averaging result time:" | awk '{print $4}')

    # Check if extraction was successful and write to file
    if [[ -n "$execution_time" ]]; then
        echo "$num_cores, $execution_time" >> "$output_file"
        echo "Cores: $num_cores, Time: $execution_time seconds"
    else
        echo "$num_cores, 0.0" >> "$output_file"
        echo "Error: Could not extract execution time for $num_cores processes."
    fi
done

echo "Execution times saved to $output_file"
