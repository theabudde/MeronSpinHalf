#!/bin/bash

#SBATCH -n 1
#SBATCH --time=99:00:00
#SBATCH --job-name="1000_no_field"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="./output/%j_%a.out"
#SBATCH --error="./output/%j_%a.err"
#SBATCH --open-mode=truncate

python ./main_no_field.py 2 1 1 8 1000 100000 /cluster/home/tbudde/MeronSpinHalf/Data $SLURM_ARRAY_TASK_ID
