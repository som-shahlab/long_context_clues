# Off-the-Shelf HuggingFace Models for EHRs

Guiding Questions:
1. Can we use off the shelf HuggingFace models on our structured EHR data?
2. Can we ablate the specific benefits of each contribution of models like CLMBR, MOTOR, Med-BERT, etc.?
3. What are "scaling laws" for Foundation Models for EHRs?

Side Goals:
1. Develop personal ML skills (PyTorch Lightning, Hydra, multi-GPU training, Wandb, HuggingFace, etc.)
2. Create Shah lab ML infrastructure akin to HazyResearch's `safari` repo

Plan:
1. Train several GPT-2 models at different scales, using the simplest possible patient timeline representation (the raw list of all codes in a patient's timeline)
2. Test each model on EHRSHOT, compare to CLMBR / count-based baselines

## Setup

```bash
conda create -n hf_env python=3.10
conda activate hf_env
pip3 install -r requirements.txt
pip3 install -e .
```

## How to Run

Create Carina node:
```bash
srun --partition=gpu --mem=200G --gres=gpu:4 --cpus-per-task=20 --time=48:00:00 --pty bash -i
conda activate hf_env && cd /share/pi/nigam/mwornow/tools/vscode && ./code tunnel --cli-data-dir /share/pi/nigam/mwornow/tools/vscode/tunnel/ 

# Separately, in VSCode, run the command "Remote-Tunnels: Connect to Tunnel..." and select "slurm-gpu"

```

Launch training run:
```bash
conda activate hf_env && cd /share/pi/nigam/mwornow/hf_ehr/src/hf_ehr/scripts
export WANDB__SERVICE_WAIT=300
python3 run.py data.dataloader.batch_size=4 data.dataloader.n_workers=10 trainer.devices=[0,1,2,3]

# Profiling
python3 run.py data.dataloader.batch_size=4 data.dataloader.n_workers=10 trainer.devices=[0,1,2,3] trainer.max_epochs=1 trainer.limit_train_batches=0.1 trainer.limit_val_batches=0.1
```

# Stats

GPT-2-342M (12-layer, 12-head, 768 embed) w/ batch size = 4 and 10 num_workers on 4 v100 32GBs:
* Memory = 25GB / GPU | Model alone = 4 GB / GPU
* Train time = 20 hrs / epoch

## Initial Setup

To create the simplest EHR tokenizer (every code is its own token):
```bash
python3 create_vocab.py
python3 create_tokenizer.py
```

You can generate sentences with the model using (you may change the sampling parameters in the `generate` function in `gpt2_lm.py`):
```bash
python interact.py --experiment experiments/lightning_logs/version_{date}
```


### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```
