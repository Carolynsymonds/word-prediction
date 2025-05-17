#!/bin/bash
#SBATCH -D /users/adgs896/inm706_week1/transformers   # Working directory
#SBATCH --job-name=depth-estimation
#SBATCH --partition=prigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -e depth.err
#SBATCH -o depth.out

# Enable flight environment
source /opt/flight/etc/setup.sh
flight env activate gridware

# Load CUDA (adjust version if needed)
module load libs/nvidia-cuda/11.2.0/bin

## Activate your virtual environment
#source /users/adgs896/inm706_week1/inm706_1/bin/activate

# Debug: confirm correct Python environment is active
echo "Using Python from: $(which python)"
python --version


export https_proxy=https://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128

export TORCH_HOME=/mnt/data/public/torch

#export WANDB_API_KEY={5219989d7c22766df2c38733ffbee9b2bfc54f80}
#export WANDB_INIT_TIMEOUT=1200

#echo $WANDB_API_KEY
#wandb login $WANDB_API_KEY --relogin


# Run the script
python -u main-keras.py