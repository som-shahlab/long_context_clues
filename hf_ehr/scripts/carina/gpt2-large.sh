#!/bin/bash
#SBATCH --job-name=gpt2-large
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-large_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/gpt2-large_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=20
#SBATCH --partition=nigam-a100
#SBATCH --gres=gpu:2

source base.sh

python3 ../run.py \
    +models=gpt2 \
    data.dataloader.batch_size=4 \
    trainer.accumulate_grad_batches=4 \
    data.dataloader.n_workers=10 \
    trainer.devices=[0,1] \
    model.config_kwargs.n_layer=36 \
    model.config_kwargs.n_head=20 \
    model.config_kwargs.n_embd=1280 \
    trainer.is_use_bf16=True \
    optimizer.lr=1e-4 \
    scheduler.num_warmup_steps=50000 \
    main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-large/ \
    logging.wandb.name=gpt2-large

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
# python3 run.py \
#     +models=gpt2 \
#     trainer.distributed_backend=deepspeed_stage_2 \
#     data.dataloader.batch_size=1 \
#     trainer.accumulate_grad_batches=16 \
#     data.dataloader.n_workers=10 \
#     trainer.devices=[0,1,2,3] \
#     trainer.is_use_fp16=True \
#     optimizer.lr=1e-4 \
#     scheduler.num_warmup_steps=50000 \
#     model.config_kwargs.n_layer=36 \
#     model.config_kwargs.n_head=20 \
#     model.config_kwargs.n_embd=1280 \
#     main.path_to_output_dir=/share/pi/nigam/mwornow/hf_ehr/cache/runs/gpt2-large-deepspeed/ \
#     logging.wandb.name=gpt2-large-deepspeed