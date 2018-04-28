#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --job-name=sample_run
#SBATCH --mail-type=END,BEGIN
#SBATCH --mail-user=mp4504@nyu.edu
#SBATCH --output=log/sbatch_log.out

module purge
module load tensorflow/python3.5/1.1.0 nltk/python3.5/3.2.4 numpy/python3.5/intel/1.13.1

python3 main.py AttentionSumReader --args_file weights/args.json --train 1
 
