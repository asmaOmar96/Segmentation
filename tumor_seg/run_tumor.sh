#!/bin/bash
#SBATCH --account=rrg-asarkar
#SBATCH --gres=gpu:a100:4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=24:0:0
#SBATCH --mail-user=asmaomar@cmail.carleton.ca
#SBATCH --mail-type=ALL
##SBATCH --mail-type=END


cd /home/asma96/projects/rrg-asarkar/asma96/tumor_seg/
module load python/3.11
module load cuda
module load gcc
module load opencv
source ~/envs/seg_env/bin/activate

#echo "GPU status at job start:"
srun nvidia-smi

# main file
srun python main.py 

## TRAINING
srun python train1.py 

# EVALUATION
#srun python evaluate.py

# testing
#python test.py

#echo "GPU status at job end:"
#nvidia-smi

