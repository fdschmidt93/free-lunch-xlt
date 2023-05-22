#!/bin/bash
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate trident_xtreme
# for SEED in 42 43 44 45 46

# env HYDRA_FULL_ERROR=1 python run.py experiment=wikiann_zs seed=${1} clf_seed=${2} trainer.max_epochs=10 module.optimizer.lr=0.00002 module.model.pretrained_model_name_or_path="xlm-roberta-base" datamodule.dataloader_cfg.train.batch_size=32 datamodule.dataloader_cfg.val.batch_size=32 datamodule.dataloader_cfg.test.batch_size=32 test_after_training=true hydra.run.dir='logs/fsxlt/wikiann/body/clfseed-${clf_seed}/seed-${seed}/lr-${module.optimizer.lr}' module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-body"  +module.clf_path='./logs/fsxlt/wikiann/zs/seed-${clf_seed}/lr-2e-05/checkpoints/9-6250.ckpt' +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/'
if [[ ${4} == "null" ]]
then
    env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=wikiann_zs seed=${1} clf_seed=${2} trainer.max_epochs=10 module.optimizer.lr=${3} module.scheduler="null"  module.model.pretrained_model_name_or_path="xlm-roberta-base" hydra.run.dir='logs/fsxlt/wikiann/soup/clfseed-${clf_seed}/seed-${seed}/lr-${module.optimizer.lr}/scheduler-null/' module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-body-hparams" +module.clf_path='./logs/fsxlt/wikiann/zs/seed-${clf_seed}/lr-2e-05/checkpoints/9-6250.ckpt' +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/' 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}_scheduler=null"'
else
    env HYDRA_FULL_ERROR=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run.py experiment=wikiann_zs seed=${1} clf_seed=${2} trainer.max_epochs=10 module.optimizer.lr=${3}  module.model.pretrained_model_name_or_path="xlm-roberta-base" hydra.run.dir='logs/fsxlt/wikiann/soup/clfseed-${clf_seed}/seed-${seed}/lr-${module.optimizer.lr}/scheduler-0.1/' module._target_="src.projects.fsxlt_avg.module.TrainBody" logger.wandb.project="wikiann-zs-body-hparams" +module.clf_path='./logs/fsxlt/wikiann/zs/seed-${clf_seed}/lr-2e-05/checkpoints/9-6250.ckpt' +module.avg_ckpts='${hydra:runtime.output_dir}/checkpoints/' 'logger.wandb.name="strategy=${strategy}_seed=${seed}_clfseed=${clf_seed}_lr=${module.optimizer.lr}_epochs=${trainer.max_epochs}_scheduler=0.1"'
fi
