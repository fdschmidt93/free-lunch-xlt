#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# seed max epochs lr dim
python run.py experiment=wikiann_zs source_lang=en seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 logger.wandb.project="wikiann-zs-roberta" test_after_training=false

# python run.py experiment=wikiann_zs source_lang=en seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 logger.wandb.project="wikiann-zs-slicer-repro" hydra.run.dir='logs/wikiann_robust_base_en_repro/${seed}/${module.optimizer.lr}/none' test_after_training=false
#
# python run.py experiment=wikiann_zs_robust seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.dim=${4} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 +module.svd=true  datamodule=wikiann_zs_debug logger.wandb.project="wikiann-zs-robust-svd-lm" +train=false # '+module.module_cfg.weights_from_checkpoint.ckpt_path="./logs/wikiann_robust_base/${seed}/${module.optimizer.lr}/${module.dim}/checkpoints/9-6250.ckpt"'

# python run.py experiment=pos_zs_robust seed=${1} trainer.max_epochs=${2} module.optimizer.lr=${3} module.dim=${4} module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 +module.svd=true '+module.module_cfg.weights_from_checkpoint.ckpt_path="./logs/pos_robust_base/${seed}/${module.optimizer.lr}/${module.dim}/checkpoints/9-6650.ckpt"' datamodule=pos_zs_debug logger.wandb.project="pos-zs-robust-svd" +train=false +trainer.enable_checkpointing=false
