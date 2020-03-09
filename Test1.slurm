#!/bin/bash

#SBATCH --mail-user=hantang.zheng@vanderbilt.edu
#SBATCH --mail-type=all
#SBATCH --account=maha_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:2
#SBATCH --time=100:00:00
#SBATCH --mem=20G
#SBATCH --output=model.out

module load GCC/6.4.0-2.28 CUDA/9.0.176 OpenMPI/2.1.1
module load scipy/0.19.1-Python-3.6.3 Tkinter/.3.6.3-Python-3.6.3
module load TensorFlow/1.8.0-Python-3.6.3  Keras/2.2.0-Python-3.6.3
module load scikit-learn

source swim_venv/bin/activate

python Upper_lower_bound_estimate.py