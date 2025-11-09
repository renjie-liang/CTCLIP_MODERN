"""
配置加载和验证模块

支持：
- YAML配置文件加载
- 环境变量替换 (${VAR})
- 配置继承 (_base_)
- 类型验证
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def expand_env_vars(config: Any) -> Any:
    """
    递归展开配置中的环境变量

    支持格式:
    - ${VAR}: 必须存在的环境变量
    - ${VAR:default}: 带默认值的环境变量

    Args:
        config: 配置对象（dict, list, str等）

    Returns:
        展开后的配置
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # 匹配 ${VAR} 或 ${VAR:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace_env(match):
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(f"Environment variable '{var_name}' is not set and no default provided")

        return re.sub(pattern, replace_env, config)
    else:
        return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个配置字典

    override中的值会覆盖base中的值

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = merge_configs(result[key], value)
        else:
            # 直接覆盖
            result[key] = value

    return result


def load_yaml(yaml_path: str) -> Dict:
    """
    加载YAML文件

    Args:
        yaml_path: YAML文件路径

    Returns:
        解析后的配置字典
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config

def cast_numeric_values(config):
    def try_cast(v):
        if isinstance(v, str):
            try:
                if '.' in v or 'e' in v.lower():
                    return float(v)
                return int(v)
            except ValueError:
                return v
        elif isinstance(v, dict):
            return {k: try_cast(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [try_cast(x) for x in v]
        else:
            return v
    return {k: try_cast(v) for k, v in config.items()}



def load_config(config_path: str, allow_base: bool = True) -> Dict:
    """
    加载配置文件，支持继承和环境变量

    工作流程:
    1. 加载配置文件
    2. 如果有 _base_ 字段，先加载基础配置
    3. 合并配置
    4. 展开环境变量
    5. 验证配置

    Args:
        config_path: 配置文件路径
        allow_base: 是否允许配置继承

    Returns:
        完整的配置字典

    Example:
        >>> config = load_config("configs/experiments/baseline.yaml")
        >>> print(config['training']['learning_rate'])
        1.25e-6
    """
    config_path = Path(config_path)
    config = load_yaml(config_path)
    config = cast_numeric_values(config)

    # 处理配置继承
    if allow_base and '_base_' in config:
        base_path = config.pop('_base_')

        # 相对路径：相对于当前配置文件
        if not Path(base_path).is_absolute():
            base_path = config_path.parent / base_path

        # 递归加载基础配置
        base_config = load_config(base_path, allow_base=True)

        # 合并配置（当前配置覆盖基础配置）
        config = merge_configs(base_config, config)

    # 展开环境变量
    config = expand_env_vars(config)

    # 验证配置
    validate_config(config)

    return config


def validate_config(config: Dict) -> None:
    """
    验证配置的完整性和正确性

    检查：
    - 必需字段是否存在
    - 数值范围是否合理
    - 路径是否有效

    Args:
        config: 配置字典

    Raises:
        ValueError: 配置无效时抛出
    """
    required_sections = ['experiment', 'data', 'model', 'training', 'validation', 'checkpoint', 'logging']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # 验证实验配置
    exp = config['experiment']
    if 'name' not in exp:
        raise ValueError("experiment.name is required")

    # 验证训练配置
    training = config['training']
    required_training = ['num_epochs', 'learning_rate']
    for key in required_training:
        if key not in training:
            raise ValueError(f"training.{key} is required")

    # 验证数值范围 - 先转换为float以处理可能的字符串类型
    if training['learning_rate'] <= 0:
        raise ValueError(f"learning_rate must be positive, got {training['learning_rate']}")

    if training['num_epochs'] <= 0:
        raise ValueError(f"num_epochs must be positive, got {training['num_epochs']}")

    # 验证验证配置
    validation = config['validation']
    if 'metrics' not in validation:
        raise ValueError("validation.metrics is required")

    # 验证checkpoint配置
    checkpoint = config['checkpoint']
    if 'save_dir' not in checkpoint:
        raise ValueError("checkpoint.save_dir is required")

    # 验证日志配置
    logging = config['logging']
    if not any([logging.get('use_wandb'), logging.get('use_tensorboard'), logging.get('use_console')]):
        raise ValueError("At least one logging backend must be enabled")


def save_config(config: Dict, save_path: str) -> None:
    """
    保存配置到YAML文件

    Args:
        config: 配置字典
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def print_config(config: Dict, indent: int = 0) -> None:
    """
    美化打印配置

    Args:
        config: 配置字典
        indent: 缩进级别
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
