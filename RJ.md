slurminfo -g xujie -u
srun --partition=hpg-b200 --gres=gpu:1 --nodes=1 --cpus-per-task=4 --mem=30gb --time=1:00:00 --account=xujie --qos=xujie  --pty bash -i 
srun --partition=hpg-b200 --gres=gpu:1 --nodes=1 --cpus-per-task=32 --mem=100gb --time=1:00:00 --account=xujie --qos=xujie  --pty bash -i 


CUDA_VISIBLE_DEVICES=5 python train.py --config configs/base_config.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/base_config.yaml
