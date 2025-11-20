# Add this at the beginning of train_step() in CTCLIPTrainer.py:
import psutil
import GPUtil
import torch
def log_memory():
    # CPU memory
    mem = psutil.virtual_memory()
    print(f"CPU Memory: {mem.percent}% used ({mem.used/1e9:.2f}GB / {mem.total/1e9:.2f}GB)")

    # GPU memory
    if torch.cuda.is_available():
        for i, gpu in enumerate(GPUtil.getGPUs()):
            print(f"GPU {i}: {gpu.memoryUsed/1e3}GB / {gpu.memoryTotal/1e3}GB ({gpu.memoryUtil*100:.1f}%)")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")

# Call log_memory() at multiple points in train_step()