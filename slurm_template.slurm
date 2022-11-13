#!/bin/bash
#SBATCH --job-name=sd
#SBATCH --partition={{job_partition}}
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output={{log_path}}
#SBATCH --account={{slurm_account}}

source activate base
cd {{working_dir}}

set -o allexport && source .env
cd StableDiffusion/app
export HF_HOME={{working_dir}}/cache/HF

{{working_dir}}/venv/bin/python3 tasks/stable_diffusion.py