#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46
# env HYDRA_FULL_ERROR=1 python run.py experiment=pos_zs seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32  test_after_training=false hydra.run.dir='logs/fsxlt/pos/zs/seed-${seed}/lr-${module.optimizer.lr}'
# env HYDRA_FULL_ERROR=1 python run.py experiment=pos_zs seed=${1} module.model.pretrained_model_name_or_path="xlm-roberta-base" test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="pos-zs-avg" +module.avg_ckpts='./logs/fsxlt/pos/zs_avg/seed-${seed}/lr-${module.optimizer.lr}/checkpoints/' hydra.run.dir='logs/fsxlt/pos/zs_avg/seed-${seed}/lr-${module.optimizer.lr}'
# env HYDRA_FULL_ERROR=1 python run.py experiment=pos_zs seed=${1} trainer.max_epochs=${2} module.model.pretrained_model_name_or_path="xlm-roberta-base" test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="pos-zs-avg" +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/' hydra.run.dir='logs/fsxlt/pos/zs_avg_${trainer.max_epochs}_epochs/seed-${seed}/lr-${module.optimizer.lr}'

# ENSEMBLING w/o fix
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=pos_zs module.model.pretrained_model_name_or_path="xlm-roberta-base" train=false test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="pos-zs-avg" +module.avg_ckpts='./logs/fsxlt/pos/zs_avg/last/' 'logger.wandb.name="strategy=unfixed-avglast-seeds_seed=${seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'
#
# ENSEMBLING
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=pos_zs clf_seed=${1} trainer.max_epochs=${2} module.model.pretrained_model_name_or_path="xlm-roberta-base" train=false test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="pos-zs-avg" +module.avg_ckpts='./logs/fsxlt/pos/body/clfseed-${clf_seed}/last/' 'logger.wandb.name="strategy=avglast-dataseed_clf-seed=${clf_seed}_seed=${seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'
#
# SOUP
env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=pos_zs clf_seed=${1} trainer.max_epochs=10 strategy="all-clfseed-soup" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="pos-zs-avg" +module.avg_ckpts='./logs/fsxlt/pos/soup/clfseed-${clf_seed}/all/' train=false test_after_training=true 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'
