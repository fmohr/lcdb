# Workflow

1. Create the container image: 
```
apptainer build my_image.sif lcdb-recipe.def
```
Be sure to execute this command in the directory where `requirements.txt` is located. This will create an image based on `condaforge/mambaforge` with all requirements installed in a conda environment called `lcdb`. 

2. When running the container instance for jobs, first pull and install the latest version of `lcdb`:
```
apptainer exec -c --bind /tmp:/tmp my_image.sif /bin/bash -c "source activate /opt/conda/envs/lcdb && cd /tmp && git clone https://github.com/fmohr/lcdb.git && cd lcdb/publications/2023-neurips && pip install . && python -c 'import sys,lcdb;modulenames = set(sys.modules) & set(globals()); print([sys.modules[name] for name in modulenames])'"
```

# TODO 

2. Initialise the SQL lite database and populate it with jobs to run. For example:
```
apptainer exec -C --bind /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdbpyexp/code/publications/2023-neurips:/mnt,/tmp:/tmp my_image.sif /bin/bash -c "source activate /opt/conda/envs/lcdb && cd /tmp && git clone https://github.com/fmohr/lcdb.git && cd lcdb/publications/2023-neurips && pip install . && cd /mnt && lcdb create --workflow SVMWorkflow" 
```
this will create the SQL Lite database in the bound `/mnt` directory. 

3. I submit the following jobfile to my SLURM scheduler to schedule a job:
```
#!/bin/sh
#SBATCH --partition=general --qos=long
#SBATCH --time=24:00:00
#SBATCH --mincpus=2
#SBATCH --mem=8192
#SBATCH --job-name=lcdb5
#SBATCH --output=lcdb5.txt
#SBATCH --error=lcdb5.txt
#SBATCH --constraint=avx2
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/
srun apptainer exec -C --bind /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdbpyexp/code/publications/2023-neurips:/mnt,/tmp:/tmp my_image.sif /bin/bash -c "mkdir -p ~/.config/ && mkdir -p ~/.config/openml/&& mkdir -p /tmp/tjviering/ && mkdir -p /tmp/tjviering/openml_cache && echo 'cachedir=/tmp/tjviering/openml_cache/' > ~/.config/openml/config && source activate /opt/conda/envs/lcdb && cd /mnt && lcdb run --workflow SVMWorkflow --executor_name 5"
```

Where:
`-C --bind /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdbpyexp/code/publications/2023-neurips:/mnt,/tmp:/tmp`
ensures that the image cannot access arbitrary directories (-C), and binds the network drive where the results should be stored. In addition, a temporary folder is bound to write temporary results. 

The line 
`/bin/bash -c "mkdir -p ~/.config/ && mkdir -p ~/.config/openml/&& mkdir -p /tmp/tjviering/ && mkdir -p /tmp/tjviering/openml_cache && echo 'cachedir=/tmp/tjviering/openml_cache/' > ~/.config/openml/config`
 takes care of making sure that there is a directory for the openml cache, and that it will be used. 

Finally, `source activate /opt/conda/envs/lcdb && cd /mnt && lcdb run --workflow SVMWorkflow --executor_name 5` takes care of starting the environment and running the workflow. 
