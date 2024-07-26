#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=bin_classifier-job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=bin_classifier-logfile-%j.log
#SBATCH --error=bin_classifier-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python

# Activate conda environment
source activate diaus_1

# Define dataset and labels path
EMBEDDINGS_FILE="/scratch/snx3000/llampert/embedding_coords/new_embedding_coordinates_sud_corse__.csv"
PROTECTION_FILE="/store/sdsc/sd29/letizia/label/labels_numeric_binary_MPA_Sud_corse.csv"
HABITAT_FILE="/store/sdsc/sd29/letizia/habitat/labels_habitat_sud_corse.csv"

# Command to run the Python script
echo "Starting the training process."
export CUDA_LAUNCH_BLOCKING=1
srun -ul $HOME/miniconda3/envs/diaus_1/bin/python training_binary_classifier.py \
    --arg_log True \
    --embeddings_file $EMBEDDINGS_FILE \
    --protection_file $PROTECTION_FILE \
    --habitat_file $HABITAT_FILE \
    --sequence_length 300 \
    --sample_subset_size 500 \
    --num_classes 2 \
    --batch_size 32 \
    --token_emb_dim 8 \
    --sample_repr_dim 128 \
    --sample_emb_dim 64 \
    --initial_learning_rate 1e-3 \
    --max_epochs 2 \
    --accelerator gpu

echo "Training completed successfully."
