#!/bin/bash
#SBATCH -J hello_world
#SBATCH --partition=testing
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.10
# or conda activate myenv
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888

