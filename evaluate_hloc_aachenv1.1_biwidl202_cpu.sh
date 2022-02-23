#!/bin/bash

#SBATCH  --output=/scratch_net/biwidl202/simomaur/log/hloc/aachen/%j.out
#SBATCH  --mem=40G
#SBATCH  --cpus-per-task=16

#source /scratch_net/biwidl202/simomaur/opt/conda/bin/conda shell.bash hook
source /scratch_net/biwidl202/simomaur/opt/conda/etc/profile.d/conda.sh
conda activate hloc
python3 -u hloc.pipelines.Aachen_v1_1.pipeline "$@"
