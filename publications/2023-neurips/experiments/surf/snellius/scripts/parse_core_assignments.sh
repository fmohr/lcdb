#!/bin/bash
# parse_core_assignments.sh
# Parses core_assignments.csv and bins datasets by resource requirements
# Usage: source parse_core_assignments.sh
# Requires: LCDB_WORKFLOW and CPUS_PER_NODE to be set

# Validate required variables
if [[ -z "$LCDB_WORKFLOW" ]]; then
    echo "Error: LCDB_WORKFLOW not set. Source config.sh first." >&2
    exit 1
fi

if [[ -z "$CPUS_PER_NODE" ]]; then
    echo "Error: CPUS_PER_NODE not set. Define in config.sh first." >&2
    exit 1
fi

# Find core_assignments.csv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# if densenn and campain is tabarena-cpu, look for core_assignments_cpu.csv
if [[ "$LCDB_WORKFLOW" == "lcdb.workflow.keras.DenseNNWorkflow" && "$CAMPAIGN_NAME" == "tabarena-cpu" ]]; then
    CORE_ASSIGNMENTS_CSV="$SCRIPT_DIR/core_assignments_cpu.csv"
else
    CORE_ASSIGNMENTS_CSV="$SCRIPT_DIR/core_assignments_gpu.csv"
fi
# CORE_ASSIGNMENTS_CSV="$SCRIPT_DIR/core_assignments.csv"

if [[ ! -f "$CORE_ASSIGNMENTS_CSV" ]]; then
    echo "Error: core_assignments.csv not found at $CORE_ASSIGNMENTS_CSV" >&2
    exit 1
fi

# Parse core_assignments.csv for this workflow
declare -a ALL_OPENML_IDS=()
declare -A OPENML_PARALLEL_TASKS=()
declare -A OPENML_CORES_PER_TASK=()
declare -A OPENML_MEMORY_PER_TASK=()

while IFS=, read -r workflow_col openmlid parallel cores memory || [[ -n "$workflow_col" ]]; do
    # Skip header
    if [[ "$workflow_col" == "workflow" ]]; then
        continue
    fi

    # Trim whitespace and carriage returns
    workflow_col="$(echo "$workflow_col" | tr -d '\r' | xargs)"
    openmlid="$(echo "$openmlid" | tr -d '\r' | xargs)"
    parallel="$(echo "$parallel" | tr -d '\r' | xargs)"
    cores="$(echo "$cores" | tr -d '\r' | xargs)"
    memory="$(echo "$memory" | tr -d '\r' | xargs)"

    # Skip empty lines
    [[ -z "$workflow_col" ]] && continue

    # Only process rows for this workflow
    if [[ "$workflow_col" == "$LCDB_WORKFLOW" ]]; then
        ALL_OPENML_IDS+=("$openmlid")
        OPENML_PARALLEL_TASKS["$openmlid"]="$parallel"
        OPENML_CORES_PER_TASK["$openmlid"]="$cores"
        OPENML_MEMORY_PER_TASK["$openmlid"]="$memory"
    fi
done < "$CORE_ASSIGNMENTS_CSV"

if (( ${#ALL_OPENML_IDS[@]} == 0 )); then
    echo "Error: No datasets found for workflow '$LCDB_WORKFLOW' in $CORE_ASSIGNMENTS_CSV" >&2
    exit 1
fi

# Group datasets by resource requirements (nodes-ntasks-cpus)
declare -A RESOURCE_BINS

for id in "${ALL_OPENML_IDS[@]}"; do
    parallel="${OPENML_PARALLEL_TASKS[$id]}"
    cores="${OPENML_CORES_PER_TASK[$id]}"

    # Calculate total cores and nodes needed
    total_cores=$((parallel * cores))
    nodes=$(( (total_cores + CPUS_PER_NODE - 1) / CPUS_PER_NODE ))
    if (( nodes < 1 )); then
        nodes=1
    fi

    # Create resource bin key: "nodes-ntasks-cpus"
    bin_key="${nodes}-${parallel}-${cores}"

    # Add this dataset to the bin
    if [[ -z "${RESOURCE_BINS[$bin_key]:-}" ]]; then
        RESOURCE_BINS[$bin_key]="$id"
    else
        RESOURCE_BINS[$bin_key]="${RESOURCE_BINS[$bin_key]},$id"
    fi
done

# Export for use by run_wrapper
export ALL_OPENML_IDS
export OPENML_PARALLEL_TASKS
export OPENML_CORES_PER_TASK
export OPENML_MEMORY_PER_TASK
export RESOURCE_BINS
