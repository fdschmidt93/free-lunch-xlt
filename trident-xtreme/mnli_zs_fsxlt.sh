#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46

# env HYDRA_FULL_ERROR=1 python run.py experiment=mnli seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  test_after_training=false hydra.run.dir='logs/fsxlt/nli/zs/seed-${seed}_lr-${module.optimizer.lr}' logger.wandb.project="mnli-zs-fsxlt-avg"
# env HYDRA_FULL_ERROR=1 python run.py experiment=mnli module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  logger.wandb.project="mnli-zs-fsxlt-avg" +module.module_cfg.weights_from_checkpoint.ckpt_path=./logs/fsxlt/nli/zs/avg-last-10.ckpt

# env HYDRA_FULL_ERROR=1 python run.py experiment=mnli seed=${1} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="mnli-zs-fsxlt-avg" +module.avg_ckpts='./logs/fsxlt/nli/zs_/seed-${seed}/lr-${module.optimizer.lr}/checkpoints/' hydra.run.dir='logs/fsxlt/nli/zs_/seed-${seed}/lr-${module.optimizer.lr}'

# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=mnli clf_seed=${1} strategy="avgall-clfseed" module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  train=false test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="mnli-zs-avg" +module.avg_ckpts='./logs/fsxlt/nli/body/clfseed-${clf_seed}/all/' #'./logs/fsxlt/nli/zs_/all/' 

# ENSEMBLE UNFIXED
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=mnli seed=${1} trainer.max_epochs=${2} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  train=false test_after_training=true logger.wandb.project="mnli-zs-fsxlt-avg" module._target_="src.projects.fsxlt_avg.module.TrainBody" +module.avg_ckpts='./logs/fsxlt/nli/zs_/last/' 'logger.wandb.name="strategy=unfixed-avglast-seeds_seed=${seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'

# SOUP
env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=mnli clf_seed=${1} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  train=false test_after_training=true logger.wandb.project="mnli-zs-avg" module._target_="src.projects.fsxlt_avg.module.TrainBody" +module.avg_ckpts='./logs/fsxlt/nli/soup/clfseed-${clf_seed}/last/' 'logger.wandb.name="strategy=last-clfseed-soup_seed=${seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'
