# Learning Curves Database for Hyperparameter Optimization

## Installation

The `install/` directory contains the files required for installation.

From this directory

```console
pip install -e "."
```

### SVM

...

### Neural Networks


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
