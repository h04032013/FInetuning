#!/bin/bash
#SBATCH --job-name=h_train_fsdp
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # one task per GPU
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4              # adjust to your node’s GPU count
#SBATCH --time=5-0:00:00
#SBATCH --mem=256G
#SBATCH --output=train_output.out
#SBATCH --error=train_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
# Optional NCCL sanity defaults for single-node:
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1              
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project
module purge 
module load Mambaforge
module load cuda cudnn
mamba activate env7

torchrun --standalone --nproc_per_node=4 fsdp_finetune.py
