#!/bin/bash

#SBATCH -n 1
#SBATCH --time=3-0
#SBATCH --job-name=scaling_analysis
#SBATCH --output=scaling_analysis.out
#SBATCH --error=scaling_analysis.err

python main_no_field.py 2 1 10 8 "$0" 200000 ./Data
