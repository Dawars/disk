#!/bin/bash

num_gpus=$1
num_nodes=$2
gpu_constraint=$3
array=$4

all_args=("$@")
rest_args=("${all_args[@]:4}")
echo "${rest_args[@]}"  # other commands


echo Num gpus: $num_gpus #num gpus
echo Nume nodes: $num_nodes # num nodes
echo GPU constraint $gpu_constraint
echo Array: $array  # 0

export MKL_DEBUG_CPU_TYPE=5  # for AMD cpus

# Conditional sbatch command based on gpu_constraint
if [ "$gpu_constraint" = "80" ]; then
    flags="--constraint=a100_80gb"
elif [ "$gpu_constraint" = "40" ]; then
    flags="--constraint=a100_40gb"
else
    flags=""
fi

set -x -e


sbatch --array=0-$array%1 $flags --ntasks-per-node $num_gpus --gres=gpu:$num_gpus -N $num_nodes run_train_sbatch.sh  \
      --num-gpus $num_gpus --num-nodes $num_nodes  "${rest_args[@]}"


#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --job-name=disk
#SBATCH --output=slurm-%x.%A.out
#SBATCH --mem=150GB
#SBATCH --partition=gpu,gpu-test,vis,vascunet
#SBATCH --gres=gpu:1
##SBATCH --constraint=a100_80gb
##SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=0-20%1

