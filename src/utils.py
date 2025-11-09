import time
import logging
import os
from collections import defaultdict
import json
import sys
import pandas as pd

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))



def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=True, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def get_logger(dir, title, level='INFO'):
    """
    Create a logger with both console and file output
    
    Args:
        dir: Directory to save log files
        title: Title/name for the log file
        level: Logging level (default: 'INFO')
    
    Returns:
        logger: Configured logger instance
    """
    os.makedirs(dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, f"{timestamp}_{title}.log")
    
    # Create logger with unique name to avoid conflicts
    logger_name = f"{title}_{timestamp}"
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # Enhanced format with timestamp
    ENHANCED_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(ENHANCED_FORMAT, DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger



def get_logger_mGPU(dir, tile, level='INFO', rank=0):
    os.makedirs(dir, exist_ok=True)
    log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, f"{log_file}_{tile}_rank{rank}.log")  # 按 rank 区分日志文件

    logger = logging.getLogger(f"rank_{rank}")  # 不同进程使用不同的 logger 名称
    logger.setLevel(level)

    # 避免重复添加 Handler
    if not logger.handlers:
        formatter = logging.Formatter("%(levelname)s:%(message)s")
        
        # 控制台日志（仅 rank=0 打印，避免重复输出）
        if rank == 0:
            chlr = logging.StreamHandler()
            chlr.setFormatter(formatter)
            logger.addHandler(chlr)
        
        # 文件日志（所有进程独立写入）
        fhlr = logging.FileHandler(log_file)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    return logger


def get_logger_file():
    import __main__
    log_name = os.path.splitext(os.path.basename(__main__.__file__))[0]
    logger = get_logger('log', log_name)
    return logger


def set_pandas_display():
    import pandas as pd
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 10000)
    pd.options.mode.chained_assignment = None
    pd.set_option('display.float_format', '{:.2f}'.format)

SET_PANDAS_DISPLAY = set_pandas_display()

class Meter:
    def __init__(self):
        self.reset()

    def add(self, value):
        self.values.append(value)

    def mean(self):
        return sum(self.values) / len(self.values) if self.values else 0.0

    def std(self):
        if len(self.values) < 2:
            return 0.0
        mean_val = self.mean()
        variance = sum((x - mean_val) ** 2 for x in self.values) / (len(self.values) - 1)
        return variance ** 0.5

    def reset(self):
        self.values = []



class MetricsManager:
    def __init__(self):
        self.meters = defaultdict(Meter)

    def __getitem__(self, key):
        return self.meters[key]

    def reset(self):
        for meter in self.meters.values():
            meter.reset()



class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_value_counts(logger, df, column_name):
    value_counts = df[column_name].value_counts()
    percentages = value_counts / value_counts.sum() * 100

    logger.info(f'{column_name}')
    for index, count in value_counts.items():
        logger.info(f'{index}: {count} ({percentages[index]:.2f}%)')


def log_number(logger, df, text):
    logger.info("---------------------------------")
    logger.info(f'{text}')
    logger.info(f"#Samples        : {len(df)}")
    logger.info(f"#CT-Note Pairs  : {len(set(df['DEID_ORDER_KEY']))}")
    logger.info(f"#Patients       : {len(set(df['deid_patient_key']))}")
