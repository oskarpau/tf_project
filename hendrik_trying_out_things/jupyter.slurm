#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=jupyter_%J.log

module load python/3.10
# or conda activate myenv
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
