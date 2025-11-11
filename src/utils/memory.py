import torch
import psutil
import os


def get_gpu_memory():
    """
    Get GPU memory usage (actual used memory, not just allocated)

    Returns:
        str: Formatted GPU memory usage string
    """
    if not torch.cuda.is_available():
        return "GPU: N/A"

    device_count = torch.cuda.device_count()
    gpu_info = []

    for i in range(device_count):
        # Use mem_get_info() for actual memory usage (more accurate than memory_allocated)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        used_mem = total_mem - free_mem

        used_gb = used_mem / 1024**3  # GB
        total_gb = total_mem / 1024**3  # GB
        percent = (used_mem / total_mem) * 100

        gpu_info.append(f"GPU{i}: {used_gb:.2f}/{total_gb:.2f}GB ({percent:.1f}%)")

    return " | ".join(gpu_info)


def get_process_memory():
    """
    Get current process RAM usage (this training process only)

    Returns:
        str: Formatted process memory usage string
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / 1024**3  # Resident Set Size in GB

    return f"Process RAM: {rss_gb:.2f}GB"


def get_system_memory():
    """
    Get system RAM usage (all processes on the node)

    Returns:
        str: Formatted system memory usage string
    """
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    total_gb = mem.total / 1024**3
    percent = mem.percent

    return f"System RAM: {used_gb:.2f}/{total_gb:.2f}GB ({percent:.1f}%)"


def get_memory_info():
    """
    Get GPU and process memory info (process-specific, not system-wide)

    Returns:
        str: Combined memory info string
    """
    gpu_mem = get_gpu_memory()
    proc_mem = get_process_memory()

    return f"{gpu_mem} | {proc_mem}"
