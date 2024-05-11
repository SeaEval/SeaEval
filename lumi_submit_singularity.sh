#!/bin/bash
#SBATCH --job-name=exampleJob
#SBATCH --account=project_462000514
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g

echo "=== Starting the job ==="
echo "MODEL_NAME is $1"
echo "BZ is $2"
echo "PROMPT_INDEX is $3"
echo "DATASET is $4"
echo "EVAL_MODE is $5"
echo "OVERWRITE is $6"

echo "=== Running the singularity container ==="



srun singularity exec -B /scratch/project_462000514:/scratch/project_462000514 /scratch/project_462000514/wangbin/workspaces/container/singularity_seaeval.sif bash run_singularity.sh $1 $2 $3 $4 $5 $6
