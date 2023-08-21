#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=168:00:00
#SBATCH --mincpus=2
#SBATCH --mem=12000
#SBATCH --job-name=lcdbM
#SBATCH --output=lcdbM%a.txt
#SBATCH --error=lcdbM%a.txt
#SBATCH --array=1-146
ulimit -n 8000
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdbpyexp/code/publications/2023-neurips/
rsync openml_cache /tmp/tjviering/ -r -v --ignore-existing
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/
srun apptainer exec -c --bind /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdbpyexp/code/publications/2023-neurips:/mnt,/tmp:/tmp test6_re2.sif /bin/bash -c "mkdir -p ~/.config/ && mkdir -p ~/.config/openml/ && echo 'cachedir=/tmp/tjviering/openml_cache/' > ~/.config/openml/config && source activate /opt/conda/envs/lcdb && pip install py_experimenter==1.2 pynisher && mkdir -p /tmp/tjviering/ && mkdir -p /tmp/tjviering/${SLURM_ARRAY_TASK_ID} && rm -rf /tmp/tjviering/${SLURM_ARRAY_TASK_ID}/lcdb && cd /tmp/tjviering/${SLURM_ARRAY_TASK_ID} && git clone https://github.com/fmohr/lcdb.git && source activate /opt/conda/envs/lcdb && cd lcdb/publications/2023-neurips && pip install . && cd /mnt && ~/.local/bin/lcdb run --config config/knn_medium.cfg --executor-name B{$SLURM_ARRAY_TASK_ID}"
