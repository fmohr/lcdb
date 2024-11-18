#!/bin/bash

set -e

mkdir -p figures

echo "Plotting performance curves..."
python plot_traj.py

echo "Plotting pareto fronts..."
python plot_pf.py