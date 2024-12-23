#!/bin/bash
#PBS -N wb_seaeval
#PBS -l select=1:ncpus=32:ngpus=2:mem=1887gb:container_engine=enroot
#PBS -l walltime=120:00:00
#PBS -j oe
#PBS -k oed
#PBS -q normal
#PBS -P 13003558
#PBS -l container_image=/data/projects/13003558/wangb1/workspaces/containers/customized_containers/seaeval_v3.sqsh
#PBS -l container_name=seaeval_v3
#PBS -l enroot_env_file=/data/projects/13003558/wangb1/workspaces/MERaLiON-AudioLLM/scripts/nscc2/env.conf
#PBS -o /data/projects/13003558/wangb1/workspaces/SeaEval/logs/q_job.out
#PBS -e /data/projects/13003558/wangb1/workspaces/SeaEval/logs/q_job.err

# HF
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/project/cache/huggingface_cache
NLTK_DATA="/project/cache/nltk_data"


enroot start \
	-r -w \
	-m /data/projects/13003558/wangb1/workspaces:/project \
	-e NLTK_DATA=$NLTK_DATA \
	-e HF_HOME=$HF_HOME \
    -e HF_ENDPOINT=$HF_ENDPOINT \
	seaeval_v3 \
	bash -c "
    cd /project/SeaEval
    pwd
    ls
    bash scripts/nscc2/nscc2_seaeval.sh $DATASET_NAME $MODEL_NAME > ${PBS_JOBID}.log 2>&1
    "
    
    
    