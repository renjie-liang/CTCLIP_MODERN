"""
Config loading and validation
"""

import os
import re
from pathlib import Path
from typing import Any, Dict
import yaml


def expand_env_vars(config: Any) -> Any:
    """Recursively expand environment variables in config"""
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
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
                raise ValueError(f"Environment variable '{var_name}' not set")

        return re.sub(pattern, replace_env, config)
    else:
        return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two config dicts"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_yaml(yaml_path: str) -> Dict:
    """Load YAML file"""
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def load_config(config_path: str, allow_base: bool = True) -> Dict:
    """
    Load config file with inheritance and env var support

    Args:
        config_path: Path to config file
        allow_base: Whether to allow config inheritance

    Returns:
        Complete config dict
    """
    config_path = Path(config_path)
    config = load_yaml(config_path)

    # Handle config inheritance
    if allow_base and '_base_' in config:
        base_path = config.pop('_base_')

        if not Path(base_path).is_absolute():
            base_path = config_path.parent / base_path

        base_config = load_config(base_path, allow_base=True)
        config = merge_configs(base_config, config)

    # Expand environment variables
    config = expand_env_vars(config)

    # Validate
    validate_config(config)

    return config


def validate_config(config: Dict) -> None:
    """Validate config completeness"""
    required_sections = ['experiment', 'data', 'model', 'training', 'validation', 'checkpoint', 'logging']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate experiment
    exp = config['experiment']
    if 'name' not in exp:
        raise ValueError("experiment.name is required")

    # Validate training (support both step-based and epoch-based)
    training = config['training']
    if 'learning_rate' not in training:
        raise ValueError("training.learning_rate is required")

    if training['learning_rate'] <= 0:
        raise ValueError(f"learning_rate must be positive")

    # Either max_steps or max_epochs must be set
    max_steps = training.get('max_steps')
    max_epochs = training.get('max_epochs')

    if max_steps is None and max_epochs is None:
        raise ValueError("Either training.max_steps or training.max_epochs must be set")

    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be positive (got {max_steps})")

    if max_epochs is not None and max_epochs <= 0:
        raise ValueError(f"max_epochs must be positive (got {max_epochs})")

    # Validate validation (support both step-based and epoch-based)
    validation = config['validation']
    if 'metrics' not in validation:
        raise ValueError("validation.metrics is required")

    # Either eval_every_n_steps or eval_every_n_epochs must be set
    eval_steps = validation.get('eval_every_n_steps')
    eval_epochs = validation.get('eval_every_n_epochs')

    if eval_steps is None and eval_epochs is None:
        raise ValueError("Either validation.eval_every_n_steps or validation.eval_every_n_epochs must be set")

    # Validate checkpoint
    checkpoint = config['checkpoint']
    if 'save_dir' not in checkpoint:
        raise ValueError("checkpoint.save_dir is required")

    # Validate logging
    logging = config['logging']
    if not any([logging.get('use_wandb'), logging.get('use_tensorboard'), logging.get('use_console')]):
        raise ValueError("At least one logging backend must be enabled")
