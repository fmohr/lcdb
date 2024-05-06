#!/bin/bash

# Function to get runtime and job state for a specific job ID
get_runtime_and_state() {
    job_id="$1"
    # Fetching runtime
    runtime=$(sacct -j "$job_id" --format=Elapsed | awk 'NR==3{print $1}')
    # Fetching job state
    job_state=$(sacct -j "$job_id" --format=State | awk 'NR==3{print $1}')
    
    # Check if the job state contains "OUT_OF_ME"
    if [[ $(sacct -j "$job_id" --format=State) == *"OUT_OF_ME"* ]]; then
        job_state="OUT_OF_MEMORY"
    fi
    
    echo "$runtime,$job_state"
}

# Add header OPENML_ID to the CSV file
sed -i '1i OPENML_ID' datasets_to_test.csv

# Add an ID column in ascending order starting from 0
awk 'BEGIN {print "ID"} {print NR-1}' datasets_to_test.csv > temp && paste -d "," temp datasets_to_test.csv > temp2 && mv temp2 datasets_to_test.csv && rm temp

# Add another column with respective runtime and state for every job
IFS=',' # Set the delimiter to comma
while IFS=',' read -r dataset_id _; do
    job_id=$(echo "6101712_$dataset_id" | cut -d ',' -f1)
    runtime_and_state=$(get_runtime_and_state "$job_id")
    echo "$runtime_and_state">> temp_runtime_and_state
done < datasets_to_test.csv

# Overwrite runtime_and_state.log to add "runtime,state" header after the initial newline
{ echo -n "runtime,state"; cat temp_runtime_and_state; } > runtime_and_state.log

# Use the paste command to concatenate datasets.csv and the modified runtime_and_state.log
paste -d ',' datasets_to_test.csv runtime_and_state.log > combined.csv

# Clean up temporary file if no longer needed
rm temp_runtime_and_state
