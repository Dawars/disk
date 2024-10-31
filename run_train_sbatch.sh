#!/usr/bin/bash
#SBATCH -N 1 # n nodes
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
##SBATCH --time 0-02:00:00
#SBATCH --job-name=disk
#SBATCH --output=slurm-%x.%A.out
#SBATCH --mem=150GB
#SBATCH --partition=gpu,gpu-test,vis,vascunet
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90


source ~/.bashrc

set -x -e
export PYTHONPATH="$PROJECTS_PATH/disk:$PROJECTS_PATH:$PROJECTS_PATH/mast3r"

# colmap_py3.10-torch2.3.1-cu12.1.sif colmap_py3.10-torch2.5.0-cu12.4.sif
srun singularity exec --nv  -B $DISK_OUT_PATH,$DISK_DATA_PATH $DOCKER_IMAGE_PATH/colmap_py3.10-torch2.5.0-cu12.4.sif \
python train.py $DISK_DATA_PATH --save-dir $DISK_OUT_PATH  $@
