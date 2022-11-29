#!/bin/bash

#SBATCH -n 1
#SBATCH --time=99:00:00
#SBATCH --job-name="1000_no_field"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="./output/%j_%a.out"
#SBATCH --error="./output/%j_%a.err"
#SBATCH --open-mode=truncate

python ./main_massless_spin_half.py 2 1 1 8 100 100000 "/cluster/home/tbudde/MeronSpinHalf/SpinHalfData" "$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID"