# The Unreasonable Effectiveness Of Early Discarding After One Epoch In Neural Network Hyperparameter Optimization

Here we provide documentation about how to reproduce experiments to compare different early discarding strategies: $i$-Epoch, $\rho$-LCE, and $r$-SHA.

In the `../learning-curves/` folder, we provide the Python notebook used to plot learning curves.

## Simplified reproducible example

The full set of experiments presented in the paper required a few steps which can be complex to reproduce.

Here we provide a relatively fast and simplified reproducible example for the $i$-Epoch and $r$-SHA stoppers on the Slice Localization regression tasks.

First, create and activate a new conda environment:
```console
conda create -n oneEpoch python=3.11 -y
conda activate oneEpoch
```

Then, move to the `example/` folder and run the following commands:

```console
./0-install.sh
./1-run.sh
./2-plot.sh
```

The data results should be located in `example/output` and the plots should be located in `example/figures`.


## More details about the full set of experiments

### Installation

A few dependencies were used for our experiments. Some can be optionally installed depending on the aspects that want to be reproduced.

Start by creating a new conda environment:

```console
conda create -n dhenv python=3.11 -y
conda activate dhenv
```

To install, `deephyper` (the software providing hyperparameter optimization and early discarding methods), at the time of the last test the commit used was `2609c9188970925bed7533b08d241fafb9a27014`:
```console
pip install "deephyper[hps,jax-cpu] @ git+https://github.com/deephyper/deephyper@develop"
```

To install `deephyper_benchmark` (the software providing learning curve benchmarks for hyperparameter optimization), at the time of the last test the commit used was `7b60adfb3e98515153ef1f82d8030aaa3c92185e`:
```console
pip install -e "git+https://github.com/deephyper/benchmark.git@main#egg=deephyper-benchmark"
```

To install `HPOBench/tabular` (the 4 regression problems):
```console
python -c "import deephyper_benchmark as dhb; dhb.install('HPOBench/tabular');"
```

To install `lcdb` (the software developed and used to generate learning curves):
```console
git clone https://github.com/fmohr/lcdb.git
cd lcdb/
git checkout bab98c915dac4d99c9e21ff68b8bcd697511f14d
cd publications/2023-neurips/
pip install -e "."
```

To install `dhexp` (the sofware developed to automate our experiments):
```
pip install -e "."
```

### Generation of classification learning curves

The learning curves were generated on the Polaris supercomputer at the Argonne Leadership Computing facility using the following scripts.

For the installation:
```console
cd ../2023-neurips/
mkdir build && cd build/
../install/polaris.sh
```

For the experiments:
```console
cd ../2023-neurips/experiments/alcf/polaris/densenn/
qsub run.sh
```

These scripts should be adapted depending on the system on which experiments are run.
They can be used as templates.

### Experimental scripts

The experimental scripts are located at `experiments/`. For `HPOBench/tabular` tasks, the scripts are located at `experiments/hpobench_tabular`. For LCDB classification tasks, the scripts are located at `experiments/lcdb2/densenn/`.

A typical script is `experiments/lcdb2/densenn/3/sequential-random-cost.sh` which was used for experiments on learning curves generated through a dataset from OpenML with `id=3`.

For different early discarding strategies the scripts are named:
- Random sampling of hyperparameters with early discarding at constant epoch: `sequential-random-const.sh`
- Random sampling of hyperparameters with early discarding SHA: `sequential-random-sha.sh`
- Random sampling of hyperparameters with early discarding LCE and MMF4 parametric model: `sequential-random-lce-mmf4.sh`
- Random sampling of hyperparameters with early discarding PFN: `sequential-random-pfn.sh`

### Plots

The scripts to create plots are located at:
- Performance Curves for Regression tasks: `experiments/hpobench_tabular/plot_all_traj.sh`
- Pareto-Front for Regression tasks: `experiments/hpobench_tabular/plot_all_pf.sh`
- Performance Curves for Classification tasks: `experiments/lcdb2/plot_all_traj.sh`
- Pareto-Front for Classification tasks: `experiments/lcdb2/plot_all_pf.sh`