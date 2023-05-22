#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46
# env HYDRA_FULL_ERROR=1 python run.py experiment=tydiqa_zs seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=32 +datamodule.dataloader_cfg.test.batch_size=32  test_after_training=false hydra.run.dir='logs/fsxlt/tydiqa/zs/seed-${seed}/lr-${module.optimizer.lr}' logger.wandb.project="tydiqa-zs-fsxlt-avg"

# env HYDRA_FULL_ERROR=1 python run.py experiment=tydiqa_zs seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=64 +datamodule.dataloader_cfg.test.batch_size=64  train=true test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="tydiqa-zs-avg" +module.avg_ckpts='./logs/fsxlt/tydiqa/zs_avg/seed-${seed}/lr-2e-05/checkpoints/' hydra.run.dir='logs/fsxlt/tydiqa/zs_avg/seed-${seed}/lr-${module.optimizer.lr}'

# env HYDRA_FULL_ERROR=1 python run.py experiment=tydiqa_zs seed=${1} trainer.max_epochs=${2} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=64 +datamodule.dataloader_cfg.test.batch_size=64  train=true test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="tydiqa-zs-avg" +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/' hydra.run.dir='logs/fsxlt/tydiqa/zs_avg_${trainer.max_epochs}_epochs/seed-${seed}/lr-${module.optimizer.lr}'

# ENSEMBLE EVALUATION
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=tydiqa_zs clf_seed=${1} trainer.max_epochs=${2} module.model.pretrained_model_name_or_path="xlm-roberta-base" +datamodule.dataloader_cfg.train.batch_size=32 +datamodule.dataloader_cfg.val.batch_size=64 +datamodule.dataloader_cfg.test.batch_size=64  train=false test_after_training=true module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="tydiqa-zs-avg" +module.avg_ckpts='./logs/fsxlt/tydiqa/body/clfseed-${clf_seed}/last/' 'logger.wandb.name="strategy=avglast-dataseed_clf-seed=${clf_seed}_seed=${seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'
#
# ENSEMBLE EVALUATION
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=tydiqa_zs clf_seed=${1} seed=${1} trainer.max_epochs=20 strategy="avglast-dataseed" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="tydiqa-zs-avg" +module.avg_ckpts='./logs/fsxlt/tydiqa/body/epochs-20/clfseed-${clf_seed}/last/' train=false test_after_training=true 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'

# soups
# env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=tydiqa_zs clf_seed=${1} seed=${1} trainer.max_epochs=20 strategy="all-clfseed-soup" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="tydiqa-zs-avg" +module.avg_ckpts='./logs/fsxlt/tydiqa/soup/clfseed-${clf_seed}/all/' train=false test_after_training=true 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"'

env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=tydiqa_zs clf_seed=${1} seed=${1} trainer.max_epochs=20 strategy="avgall-dataseed" module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="tydiqa-zs-avg" train=false test_after_training=true 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}"' +module.module_cfg.weights_from_checkpoint.ckpt_path=./logs/fsxlt/tydiqa/body/epochs-20/clfseed-42/all.ckpt

