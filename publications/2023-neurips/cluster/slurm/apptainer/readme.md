# Workflow

1. Create the container image: 
```
apptainer build my_image.sif lcdb-recipe.def
```
This will create an image based on `condaforge/mambaforge` with the latest version of lcdb installed. The recipe file is included in this folder. 

2. Initialise the SQL lite database and populate it with jobs to run. For example:
```
apptainer exec -C --bind /tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/tjviering/lcdbpyexp/code/publications/2023-neurips:/mnt,/tmp:/tmp my_image.sif /bin/bash -c "source activate /opt/conda/envs/lcdb && cd /mnt && lcdb create --workflow SVMWorkflow" 
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

# Improved workflow (TODO)

It seems rather tedious to generate a new image each time we make a slight update to the lcdb code. Therefore, it would be better to install lcdb from scratch each time the image is booted up, but to install all dependencies already in the image beforehand. That would be really nice, but that is not working yet... Currently my attempt is:

First I export all the requirements to a `requirements.txt`:
```
aiofiles==22.1.0
aiosqlite==0.19.0
anyio==3.7.1
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
arrow==1.2.3
asttokens==2.2.1
attrs==23.1.0
Babel==2.12.1
backcall==0.2.0
beautifulsoup4==4.12.2
bleach==6.0.0
certifi==2023.5.7
cffi==1.15.1
charset-normalizer==3.1.0
cmake==3.26.4
comm==0.1.3
ConfigSpace==0.6.1
debugpy==1.6.7
decorator==5.1.1
defusedxml==0.7.1
exceptiongroup==1.1.2
executing==1.2.0
fastjsonschema==2.17.1
filelock==3.12.2
fqdn==1.5.1
idna==3.4
importlib-metadata==6.7.0
ipykernel==6.24.0
ipython==8.14.0
ipython-genutils==0.2.0
isoduration==20.11.0
jedi==0.18.2
Jinja2==3.1.2
joblib==1.3.1
json5==0.9.14
jsonpointer==2.4
jsonschema==4.17.3
jupyter-events==0.6.3
jupyter-ydoc==0.2.4
jupyter_client==8.3.0
jupyter_core==5.3.1
jupyter_server==2.7.0
jupyter_server_fileid==0.9.0
jupyter_server_terminals==0.4.4
jupyter_server_ydoc==0.8.0
jupyterlab==3.6.5
jupyterlab-pygments==0.2.2
jupyterlab_server==2.23.0
liac-arff==2.5.0
lit==16.0.6
MarkupSafe==2.1.3
matplotlib-inline==0.1.6
minio==7.1.15
mistune==3.0.1
more-itertools==9.1.0
mpmath==1.3.0
mysql-connector-python==8.0.33
nbclassic==1.0.0
nbclient==0.8.0
nbconvert==7.6.0
nbformat==5.9.0
nest-asyncio==1.5.6
networkx==3.1
notebook==6.5.4
notebook_shim==0.2.3
numpy==1.25.0
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.2.10.91
nvidia-cusolver-cu11==11.4.0.1
nvidia-cusparse-cu11==11.7.4.91
nvidia-nccl-cu11==2.14.3
nvidia-nvtx-cu11==11.7.91
openml==0.14.0
overrides==7.3.1
packaging==23.1
pandas==2.0.3
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
platformdirs==3.8.0
prometheus-client==0.17.0
prompt-toolkit==3.0.39
protobuf==3.20.3
psutil==5.9.5
ptyprocess==0.7.0
pure-eval==0.2.2
py-experimenter==1.1.0
pyarrow==12.0.1
pycparser==2.21
Pygments==2.15.1
pyparsing==3.1.0
pyrsistent==0.19.3
python-dateutil==2.8.2
python-json-logger==2.0.7
pytz==2023.3
PyYAML==6.0
pyzmq==25.1.0
requests==2.31.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
scikit-learn==1.3.0
scipy==1.11.1
Send2Trash==1.8.2
six==1.16.0
sniffio==1.3.0
soupsieve==2.4.1
stack-data==0.6.2
sympy==1.12
terminado==0.17.1
threadpoolctl==3.1.0
tinycss2==1.2.1
tomli==2.0.1
torch==2.0.1
tornado==6.3.2
tqdm==4.65.0
traitlets==5.9.0
triton==2.0.0
typing_extensions==4.7.1
tzdata==2023.3
uri-template==1.3.0
urllib3==2.0.3
wcwidth==0.2.6
webcolors==1.13
webencodings==0.5.1
websocket-client==1.6.1
xgboost==1.7.6
xmltodict==0.13.0
y-py==0.5.9
ypy-websocket==0.8.2
zipp==3.15.0
```

Then I make the following recipe:
```
Bootstrap: docker
From: condaforge/mambaforge

%files
	requirements.txt /opt

%post
    conda create -n lcdb python=3.9 -y
    exec /opt/conda/envs/lcdb/bin/pip install -r /opt/requirements.txt 
	
%runscript
	git clone https://github.com/fmohr/lcdb.git code
	cd code/publications/2023-neurips
    exec /opt/conda/envs/lcdb/bin/pip install . 
```
Somehow, the runscript errors... If I instead try to install via argument calls, I also get errors:
`apptainer exec -c --bind /tmp:/tmp test4.sif /bin/bash -c "source activate /opt/conda/envs/lcdb && git clone https://github.com/fmohr/lcdb.git /tmp/code && cd /tmp/code/publications/2023-neurips && pip install . "`
gives me 
