##!/usr/bin/bash
#BSUB -n 8
# #BSUB -n 12
# #BSUB -W 4:00
#BSUB -W 24:00
# #BSUB -W 72:00
#BSUB -R "rusage[mem=4000, scratch=1500, ngpus_excl_p=1] span[hosts=1]"
# #BSUB -R "rusage[mem=3000, scratch=800, ngpus_excl_p=2] span[hosts=1]"
# #BSUB -R "rusage[mem=3000, ngpus_excl_p=2] span[hosts=1]"
# #BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -R "select[gpu_model0==NVIDIATITANRTX]"
# #BSUB -R "select[gpu_model0==TeslaV100_SXM2_32GB]"
# #BSUB -R "select[gpu_model0==A100_PCIE_40GB]"
#BSUB -o /cluster/work/cvl/kanakism/output_logs/
#BSUB -e /cluster/work/cvl/kanakism/output_logs/

set -e

#COPYING=0
#function copyback() {
#    if [[ $COPYING -eq 0 ]]; then
#        COPYING=1
#        [[ "$VERBOSE" ]] && echo "Copying back new files from ${TMPDIR}/experiments/ to /cluster/work/cvl/kanakism/results/QuantNetBinDescr/."
#        rsync -SHParq ${TMPDIR}/experiments/* /cluster/work/cvl/kanakism/results/QuantNetBinDescr/ || ( echo "Trouble copying files back." && exit 1 )
#        [[ "$VERBOSE" ]] && echo "Done copying files."
#    fi
#


#rsync -aq ./ ${TMPDIR}
##LOGDIR=/cluster/home/kanakism/logs
##mkdir -p ${TMPDIR}/data
##mkdir -p ${TMPDIR}/saved
#mkdir -p ${TMPDIR}/experiments
##SOURCEDIR=/cluster/work/cvl/shared/mtlgroup/data
#SOURCEDIR=/cluster/work/cvl/kanakism/datasets
## tar -I pigz -xf ${SOURCEDIR}/PASCALContext.tar.gz -C ${TMPDIR}/data
##tar -I pigz -xf ${SOURCEDIR}/NYUD_vid.tar.gz -C ${TMPDIR}/data
#tar -I pigz -xf ${SOURCEDIR}/hpatches.tar.gz -C ${TMPDIR}/datasets
#tar -I pigz -xf ${SOURCEDIR}/mscoco2017.tar.gz -C ${TMPDIR}/datasets
##tar -I pigz -xf ${SOURCEDIR}/BSR.tar.gz -C ${TMPDIR}/data
##tar -I pigz -xf ${SOURCEDIR}/bdd100k.tar.gz -C ${TMPDIR}/data

#export WANDB_API_KEY="f0218a5aa8bfe83966eae0c2daed91fd8b029302"
#export WANDB_ENTITY="quantnetbindescr"

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

#IFS="," read -a DEVICE_IDS <<< $CUDA_VISIBLE_DEVICES
#NUM_GPUS=${#DEVICE_IDS[@]}

#export EXP_HOME=${TMPDIR}/saved
#export DATA_HOME=${TMPDIR}/data
## optional, for pretrained models
#export TORCH_HOME=/cluster/work/cvl/brdavid/pretrained

#source /cluster/home/kanakism/venvs/QuantNetBinDescr/bin/activate

# BSUB does SIGINT, SIGTERM, SIGKILL 10 s. apart
#trap copyback SIGINT SIGTERM

#if [ $NUM_GPUS -gt 1 ]; then
#    python src/main.py --cfg config/$CONFIG --trainer.gpus $NUM_GPUS --trainer.accelerator ddp &> ${LOGDIR}/${LSB_JOBNAME}.txt
#else
#    python src/main.py --cfg config/$CONFIG --trainer.gpus $NUM_GPUS &> ${LOGDIR}/${LSB_JOBNAME}.txt
#fi

detector=muri_binary
matcher=NNH-ratio0.8_dist32

source ~/venvs/hloc/bin/activate
#cd $TMPDIR
LOGFILE=/cluster/home/kanakism/logs/${detector}_${matcher}_$(date -d "today" +"%Y%m%d%H%M")_log.txt
bash << EOF >> ${LOGFILE} 2>&1

python -m hloc.pipelines.Aachen_v1_1.pipeline --outputs /cluster/work/cvl/kanakism/results/hloc/Aachen_v1_1_${detector}_${matcher} --dataset /cluster/work/cvl/kanakism/datasets/aachen_v1.1/ --num_loc 30 --detector ${detector} --matcher ${matcher}

EOF

copyback
