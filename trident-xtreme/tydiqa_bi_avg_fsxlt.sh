#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46

env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=tydiqa_bi_avg seed=${1} data_seed=${2} shots=${3} lang=${4} lang_val_len=${5} strategy="avg-all" trainer.max_epochs=10 module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" hydra.run.dir='logs/fsxlt/tydiqa/bi_avg/${lang}/${shots}/data_seed-${data_seed}/seed-${seed}/lr-${module.optimizer.lr}' 
