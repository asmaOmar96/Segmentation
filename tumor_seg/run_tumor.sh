#!/bin/bash
#SBATCH --account=def-asarkar
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:0:0
#SBATCH --mail-user=asmaomar@cmail.carleton.ca
#SBATCH --mail-type=ALL
##SBATCH --mail-type=END


cd /home/asma96/projects/def-asarkar/asma96/tumor_seg/
module load python/3.11
module load cuda
module load gcc
module load opencv
source seg_env/bin/activate

#echo "GPU status at job start:"
srun nvidia-smi

## TRAINING
#srun python tumor_seg_data.py 

# EVALUATION
#srun python evaluate.py

# testing
python test.py

#echo "GPU status at job end:"
#nvidia-smi

