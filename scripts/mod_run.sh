#!/bin/bash
#SBATCH --job-name=h_train
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --time=0-16:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
export OMP_NUM_THREADS=8

module purge
module load Mambaforge
module load cuda cudnn

mamba activate env3
cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project

# Resolve MASTER_ADDR from SLURM_NODELIST
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
export MASTER_ADDR=${nodes_array[0]}
export MASTER_PORT=$((10000 + RANDOM % 50000))
# export MASTER_PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export NODE_RANK=$SLURM_NODEID

# srun torchrun \
#   --nproc_per_node=4 \
#   --nnodes=2 \
#   --node_rank=$NODE_RANK \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   modified_finetune.py

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  modified_finetune.py
