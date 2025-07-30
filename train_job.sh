#!/bin/bash
#SBATCH --job-name=h_train
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-16:00:00
#SBATCH --mem=128G
#SBATCH --output=train_output.out
#SBATCH --error=train_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_hub"

cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project

module purge
module load Mambaforge
module load cuda cudnn

# Activate conda environment (optional)
mamba activate env4

# Run training
python get_peft_names.py
