#!/bin/bash

#SBATCH  --output=/scratch-second/simomaur/log/hloc/aachen/%j.out
#SBATCH  --constraint='titan_x|titan_xp|geforce_rtx_2080_ti'
#SBATCH  --mem=40G
#SBATCH  --cpus-per-task=4

#source /scratch_net/biwidl202/simomaur/opt/conda/bin/conda shell.bash hook
source /scratch_net/biwidl202/simomaur/opt/conda/etc/profile.d/conda.sh
conda activate hloc
python3 -u evaluate_hloc_aachen_kp2d.py "$@"
