#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46

# env HYDRA_FULL_ERROR=1 python run.py experiment=wikiann_zs seed=${1} trainer.max_epochs=${2} module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 hydra.run.dir='logs/fsxlt/wikiann/zs_${trainer.max_epochs}_epochs/seed-${seed}/lr-${module.optimizer.lr}' module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-fsxlt-avg" +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/'

# env HYDRA_FULL_ERROR=1 python run.py experiment=wikiann_zs seed=${1} trainer.max_epochs=10 module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 hydra.run.dir='logs/fsxlt/wikiann/zs_50pct_lr_schedule/seed-${seed}/lr-${module.optimizer.lr}' module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-avg-50pct-lr-schedule" +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/' module.scheduler.num_warmup_steps=0.5

# ENSEMBLE FIXED
# env HYDRA_FULL_ERROR=1 python run.py experiment=wikiann_zs clf_seed=${1} trainer.max_epochs=10 strategy="avgall-clfseed" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-fsxlt-avg" +module.avg_ckpts='./logs/fsxlt/wikiann/body_/clfseed-${clf_seed}/all/' train=false

# ENSEMBLE UNFIXED
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=wikiann_zs trainer.max_epochs=10 strategy="unfixed-avgall-seeds" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-fsxlt-avg" +module.avg_ckpts='./logs/fsxlt/wikiann/zs/last/' train=false test_after_training=true 'logger.wandb.name="strategy=unfixed-avglast-seeds_seed=${seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'

# SOUP
env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=wikiann_zs clf_seed=${1} trainer.max_epochs=10 strategy="last-clfseed-soup" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-fsxlt-avg" +module.avg_ckpts='./logs/fsxlt/wikiann/soup/clfseed-${clf_seed}/last/' train=false test_after_training=true 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'
