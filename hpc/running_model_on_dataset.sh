#!/bin/bash

# The name of the job is test_job
#SBATCH -J qwen3
# Format of the output filename: slurm-jobname.jobid.out
#SBATCH --output=logs/slurm-%x.%j.out
# The job requires 1 compute node
#SBATCH -N 1

# The job requires 1 task per node
#SBATCH --ntasks-per-node=1

# The maximum walltime of the job is 24 minutes
#SBATCH -t 48:00:00

#SBATCH --mem=32G

# If you keep the next two lines, you will get an e-mail notification
# whenever something happens to your job (it starts running, completes or fails)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oskarpau@ut.ee

# Keep this line if you need a GPU for your job
#SBATCH --partition=gpu

# Indicates that you need one GPU node
#SBATCH --gres=gpu:a100-40g

# Commands to execute go below
# Load CUDA module (for jobs that need GPU)
module load cuda/12.1

# Load Python
module load python/3.10.10
# Activate your environment
source ../../envs/tf_project/bin/activate

python ../src/run_model_on_dataset.py