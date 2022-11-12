#!/bin/bash

#SBATCH -n 1
#SBATCH --time=99:00:00
#SBATCH --job-name="$0_no_field"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="$0_no_field.out"
#SBATCH --error="$0_no_field.err"
#SBATCH --open-mode=truncate

python main_no_field.py 2 1 10 8 "$0" 200000 ./Data
