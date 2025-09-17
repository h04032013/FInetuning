#!/bin/bash
#SBATCH --job-name=h_gen
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --mem=128G
#SBATCH --output=gen_output.out
#SBATCH --error=gen_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu


cd /n/netscratch/dam_lab/Lab/hdiaz/ft_project


module purge
module load Mambaforge
module load cuda cudnn
mamba activate env8

# Run training
python main.py
