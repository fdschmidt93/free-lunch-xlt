#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46

# if [[ ${1} -eq 42 ]] && [[ ${2} -eq 42 ]] && [[ ${3} -eq 100 ]]; then
#     bash ./run_tydiqa_avg.sh
# fi
env HYDRA_FULL_ERROR=1 HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=tydiqa_multi_avg seed=${1} data_seed=${2} shots=${3} trainer.max_epochs=${4} module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" hydra.run.dir='logs/fsxlt/tydiqa/avg/epochs-${trainer.max_epochs}/${shots}/data_seed-${data_seed}/seed-${seed}/lr-${module.optimizer.lr}'  strategy="avg-all"
