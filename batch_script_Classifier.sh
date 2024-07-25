#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=classifier-job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=classifier-logfile-%j.log
#SBATCH --error=classifier-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python

# Activate conda environment
source activate diaus_1

# Define dataset and labels path
EMBEDDINGS_FILE="/path/to/embedding_coordinates_sud_corse_version.csv"
LABELS_FILE="label/ordinal_label_Sud_Corse.csv"
HABITAT_FILE="habitat/labels_habitat_sud_corse.csv"

# Command to run the Python script
echo "Starting the training process."
export CUDA_LAUNCH_BLOCKING=1

srun -ul $HOME/miniconda3/envs/diaus_1/bin/python training_Classifier.py \
    --arg_log True \
    --embeddings_file $EMBEDDINGS_FILE \
    --labels_file $LABELS_FILE \
    --habitats_file $HABITAT_FILE \
    --sequence_length 300 \
    --num_classes 4 \
    --batch_size 32 \
    --sample_emb_dim 64 \
    --habitat_emb_dim 3 \
    --initial_learning_rate 1e-3 \
    --max_epochs 50

echo "Training completed successfully."
