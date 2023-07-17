# Learning Curves Database for Hyperparameter Optimization

## Installation

The `install/` directory contains the files required for installation.

From this directory

```console
pip install -e "."
```

## Example Usage

Create a configuration file in `config/` such as `config/example.cfg`:

```
[PY_EXPERIMENTER]

provider = sqlite
database = lcdb
table = knn

# train_size and hyperparameters are omitted since they are computed automatically
keyfields = workflow:str, openmlid:int, valid_prop: float, test_prop: float, seed_outer:int, seed_inner:int, train_sizes:text, hyperparameters:text, monotonic:boolean
workflow = lcdb.workflow.sklearn.KNNWorkflow
openmlid = 15
valid_prop = 0.1
test_prop = 0.1
seed_outer = 0
seed_inner = 0
train_sizes = -1
hyperparameters = None
monotonic = 0

resultfields = result:LONGTEXT
resultfields.timestamps = false
```

Then test the workflow used in this configuration `lcdb.workflow.sklearn.KNNWorkflow` by running:

```console
lcdb test --config config/example.cfg --verbose
```

To create a database of experiments:

```console
lcdb create --config config/example.cfg
```

To pull and execute experiments from the database:

```console
lcdb run --config config/example.cfg --executor-name debug
```


## TODO

### SVM

...

### Neural Networks

Implementing regularization techniques for neural networks:

- [ ] augmentation
    - [ ] adversarial
    - [ ] cutmix
    - [ ] mixup
    - [ ] cutout
- [x] batch normalization
- [ ] stochastic weight averaging (available in torch)
- [ ] ensembling
    - [ ] ensembling with uniform weighting
    - [ ] ensembling with bootstrap resampling
- [x] residual connections
- [ ] shake shape (?)
- [x] weight decay
- [ ] shake drop
- [ ] lookahead
- [x] dropout


```console
conda create -n lcdb python=3.9 -y
conda activate lcdb
mkdir build && cd build/
git clone https://github.com/automl/Auto-PyTorch.git && cd Auto-PyTorch/

# The following line will not run on MacOS (arm64)
conda install gxx_linux-64 gcc_linux-64 swig -y

cat requirements.txt | xargs -n 1 -L 1 pip install
pip install -e "."

git checkout regularization_cocktails

cd ..
git clone https://github.com/releaunifreiburg/WellTunedSimpleNets.git
```

```
pip install "pandas<2.0.0"
pip install "smac<2.0.0"
```

**Comment**:
- Installing directly the requirements from `regularization_cocktails` is not working... the requirements are very strict (i.e., strict version is used for all dependencies) and can hardly adapt to other environments.


**Regularization Cocktails**:
- [ ] `stochastic_weight_averaging`
    - Description: If stochastic weight averaging should be used.
    - Value: `[True, False]`
- [ ] `snapshot_ensembling`
    - Description: If snapshot ensembling should be used.
    - Value: `[True, False]`
- [ ] `lookahead`
    - Description: If the lookahead optimizing technique should be used.
    - Value: `[True, False]`
- [ ] `weight_decay`
    - Description: If weight decay regularization should be used.
    - Value: `[True, False]`
- [ ] `batch_normalization`
    - Description: If batch normalization regularization should be used.
    - Value: `[True, False]`
- [ ] `skip_connection`
    - Description: If skip connections should be used. Turns the network into a residual network.- Value: `[True, False]`
- [ ] `dropout` 
    - Description: If dropout regularization should be used.
    - Value: `[True, False]`
- [ ] `multi_branch_choice`
    - Description: Multibranch network regularization. Only active when `skip_connection` is active.
    - Value: `['none', 'shake-shake', 'shake-drop']`
- [ ] `augmentation`
    - Description: If methods that augment examples should be used.
    - Value: `['mixup', 'cutout', 'cutmix', 'standard', 'adversarial']`
