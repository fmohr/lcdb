#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=2:00:00
#SBATCH --threads-per-core=1

source ~/.bashrc
conda activate lcdb

declare -a result_files=("$@")

echo "==== Starting campaign upload ===="
echo "Total result files to process: ${#result_files[@]}"

# -------------------------------------------------------------------------
# Load .env for authentication
# -------------------------------------------------------------------------
if [ -f "$ENV_PATH" ]; then
    echo "Loading environment variables from $ENV_PATH"
    export $(grep -v '^#' "$ENV_PATH" | xargs)
else
    echo "Warning: .env file not found at $ENV_PATH"
fi

# -------------------------------------------------------------------------
# Authenticate to pCloud
# -------------------------------------------------------------------------
echo "Authenticating with pCloud..."
srun lcdb pcloud -p "$ENV_PATH" || { echo "Authentication failed in lcdb pcloud, exiting."; exit 1; }

# Refresh environment (new token)
if [ -f "$ENV_PATH" ]; then
    export $(grep -v '^#' "$ENV_PATH" | xargs)
fi

# -------------------------------------------------------------------------
# Upload results
# -------------------------------------------------------------------------
uploaded=0
missing=0

for file in "${result_files[@]}"; do
    if [ -f "$file" ]; then
        echo "Uploading existing result file: $file"
    else
        echo "Warning: missing result file $file (will still upload logs)"
    fi
    srun lcdb add -c "$CAMPAIGN_NAME" -t "$PCLOUD_TOKEN" -l True "$file" || true

done