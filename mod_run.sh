#!/bin/bash
#SBATCH --job-name=h_train
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0-16:00:00
#SBATCH --mem=128G
#SBATCH --output=train_output.out
#SBATCH --error=train_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

# Set environment variables
export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"
export OMP_NUM_THREADS=8

# Load modules
module purge
module load Mambaforge
module load cuda cudnn

# Activate conda env
mamba activate env3

# Go to project directory
cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project

# Run training using torchrun
torchrun \
  --nproc_per_node=4 \
  modified_finetune.py
