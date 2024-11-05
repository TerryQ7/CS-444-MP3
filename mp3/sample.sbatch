#!/bin/bash

### Do not change the following 8 lines.
#SBATCH --time=03:00:00                         # Job run time (hh:mm:ss)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=8                     # Number of task (cores/ppn) per node
#SBATCH --mem=16g                               # Amount of CPU memory
#SBATCH --job-name=cs444mp3                     # Name of batch job
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=cs444-ic
#SBATCH --gres=gpu:A100                
### End Do not change block

#SBATCH --mail-user=hq10@illinois.edu          # replace with your email
#SBATCH --mail-type=NONE                        # Type of email notifications to send

#SBATCH --output=cs444mp3.out.%j                # Default name of batch job
# 					        # output file if not set on command line
#SBATCH --error=cs444mp3.err.%j                 # Default name of batch job 
# 					        # error file if not set on command line

###############################################################################
# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR
cd ${SLURM_SUBMIT_DIR}
###############################################################################

echo "Runnng on `hostname`"

# Load Python/Anaconda module (Enable Python in batch job environment)
module load git/2.19.0 python/3.9.16 cuda/12.4

# Activate virtual environment
source /projects/illinois/class/cs444/saurabhg/fa2024/mp3/venv/bin/activate

nvidia-smi
export PYTHONUNBUFFERED=1

# This creates output directory for every run. Change OUTPUT_DIR while
# performing each run and ablation experiment

# Assign a default value of runs/run1 to 'OUTPUT_DIR' only if not set on
# command line already. For each different experiment you run, you should use a
# different OUTPUT_DIR. Change it in the command line flag.
OUTPUT_DIR=${OUTPUT_DIR:="runs/run1"}
mkdir -p ${OUTPUT_DIR}            

LOG_FILE="${OUTPUT_DIR}/log.txt"


python demo.py --seed 1 --lr 1e-2 --batch_size 1 --output_dir ${OUTPUT_DIR} --coco_dir /projects/illinois/class/cs444/saurabhg/fa2024/mp3/coco-animal 2>&1 | tee ${LOG_FILE}
