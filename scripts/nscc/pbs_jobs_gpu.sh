#!/bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=16:ngpus=4:mem=110gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -P 13003565

GPU=0,1,2,3

################################################# 
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: executing queue is $PBS_QUEUE
echo -e "Work folder is $PWD\n\n"

echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
#################################################
cd $PBS_O_WORKDIR
echo -e "Work folder is $PWD\n\n"

#################################################
source /data/projects/13003565/wangb1/anaconda3/etc/profile.d/conda.sh
conda activate seaeval_v2
echo "Virtual environment activated"

#################################################
#################################################


#################################################

echo "=  =  =  =  =  =  =  =  =  =  =  =  ="
echo "DATASET: $DATASET"
echo "MODEL_NAME: $MODEL_NAME"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "EVAL_MODE: $EVAL_MODE"
echo "OVERWRITE: $OVERWRITE"
echo "NUMBER_OF_SAMPLES: $NUMBER_OF_SAMPLES"
echo "GPU: $GPU"
echo "=  =  =  =  =  =  =  =  =  =  =  =  ="

#################################################

mkdir -p log/$MODEL_NAME
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

# 2>&1 | tee log/$EVAL_MODE/$DATASET/${MODEL_NAME}_p${PROMPT_INDEX}.log

#################################################

echo "Finished"