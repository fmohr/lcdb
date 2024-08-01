echo "Running jobs: $(squeue -u $USER --state=Running | wc -l)"
echo "Pending jobs: $(squeue -u $USER --state=PD | wc -l)"