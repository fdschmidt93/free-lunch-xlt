#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46
# env HYDRA_FULL_ERROR=1 python run.py experiment=wikiann_multi_avg seed=${1} data_seed=${2} trainer.max_epochs=10 module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" hydra.run.dir='logs/fsxlt/wikiann/avg/${shots}/data_seed-${data_seed}/seed-${seed}/lr-${module.optimizer.lr}' shots=${3} strategy="avg-all"

env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=wikiann_multi_avg seed=${1} data_seed=${2} shots=${3} trainer.max_epochs=${4} module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" hydra.run.dir='logs/fsxlt/wikiann/avg/epochs-${trainer.max_epochs}/${shots}/data_seed-${data_seed}/seed-${seed}/lr-${module.optimizer.lr}' strategy="avg-all" 
