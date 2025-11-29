import torch
import psutil
import os
import subprocess


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


def get_gpu_utilization():
    """
    Get GPU utilization (compute usage percentage)

    Returns:
        dict: {gpu_id: utilization_percent} or empty dict if unavailable
    """
    if not torch.cuda.is_available():
        return {}

    try:
        # Use nvidia-smi to get GPU utilization
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            utilization = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) == 2:
                        gpu_id = int(parts[0].strip())
                        util = int(parts[1].strip())
                        utilization[gpu_id] = util
            return utilization
    except:
        pass

    return {}


def get_memory_info():
    """
    Get GPU and process memory info (process-specific, not system-wide)

    Returns:
        str: Combined memory info string
    """
    gpu_mem = get_gpu_memory()
    proc_mem = get_process_memory()

    return f"{gpu_mem} | {proc_mem}"


def get_detailed_gpu_memory_info():
    """
    Get detailed GPU and memory info with utilization for profiling

    Returns:
        list: List of formatted strings for each GPU + process memory
    """
    if not torch.cuda.is_available():
        return ["GPU: N/A", get_process_memory()]

    device_count = torch.cuda.device_count()
    gpu_util = get_gpu_utilization()
    info_lines = []

    for i in range(device_count):
        # Get memory info
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        used_mem = total_mem - free_mem

        used_gb = used_mem / 1024**3
        total_gb = total_mem / 1024**3
        percent = (used_mem / total_mem) * 100

        # Get utilization
        util_str = f" | Util: {gpu_util[i]}%" if i in gpu_util else ""

        info_lines.append(f"GPU {i}: {used_gb:.2f}/{total_gb:.2f}GB ({percent:.1f}%){util_str}")

    # Add process memory
    info_lines.append(get_process_memory())

    return info_lines
