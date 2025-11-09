srun --partition=hpg-b200 --gres=gpu:2 --nodes=1 --cpus-per-task=8 --mem=100gb --time=2:00:00 --account=xujie --qos=xujie  --pty bash -i 
srun --partition=hpg-b200 --gres=gpu:1 --nodes=1 --cpus-per-task=8 --mem=100gb --time=2:00:00 --account=xujie --qos=xujie  --pty bash -i 


CUDA_VISIBLE_DEVICES=0 python train.py --config configs/base_config.yaml

