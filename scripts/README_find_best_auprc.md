# Find Best AUPRC Epoch Script

这个脚本用于从wandb日志中查找AUPRC（或其他指标）最高的epoch。

## 功能特性

- **离线模式**：从本地 `./wandb` 目录读取summary文件（默认模式）
- **在线模式**：使用wandb API从服务器获取完整历史记录
- 支持自定义指标名称
- 可以分析单个run或所有runs
- 显示最佳epoch的其他相关指标

## 使用方法

### 基本用法（离线模式）

查找AUPRC最高的epoch：
```bash
python scripts/find_best_auprc_epoch.py
```

### 查看所有runs

```bash
python scripts/find_best_auprc_epoch.py --all
```

### 使用自定义指标

查找AUROC最高的epoch：
```bash
python scripts/find_best_auprc_epoch.py --metric val/macro_auroc
```

查找weighted AUPRC最高的epoch：
```bash
python scripts/find_best_auprc_epoch.py --metric val/weighted_auprc
```

### 分析特定的run

```bash
python scripts/find_best_auprc_epoch.py --run-id e6v1zxzz
```

### 在线模式（需要wandb登录）

```bash
# 首先登录wandb
wandb login

# 使用在线API获取完整历史
python scripts/find_best_auprc_epoch.py --online --project your-project-name --entity your-username
```

## 输出示例

```
Using local wandb files (offline mode)
Note: Only reading final summaries, not full history


Scanning local wandb runs...
============================================================
run-20251201_013612-e6v1zxzz: Step 9320, val/macro_auprc: 0.2795
run-20251201_002312-ngy5vz0a: Step 9320, val/macro_auprc: 0.2852
run-20251130_225338-po1owjtc: Step 9320, val/macro_auprc: 0.2795
run-20251130_161815-w3rvkxbm: Step 9320, val/macro_auprc: 0.2827
run-20251130_040406-k7l9tqgt: Step 9320, val/macro_auprc: 0.2804

============================================================
BEST RUN: run-20251201_002312-ngy5vz0a
Best Epoch/Step: 9320
Best val/macro_auprc: 0.2852
============================================================

Other metrics at this epoch:
  val/macro_auroc: 0.6269
  val/weighted_auprc: 0.3507
  val/weighted_auroc: 0.6180
  val/macro_f1: 0.3757
  val/weighted_f1: 0.4524
  val/macro_recall: 0.7849
  val/macro_precision: 0.2282
```

## 支持的指标

脚本会自动显示最佳epoch的以下指标：
- `val/macro_auroc`
- `val/weighted_auprc`
- `val/weighted_auroc`
- `val/macro_f1`
- `val/weighted_f1`
- `val/macro_recall`
- `val/macro_precision`
- `val/weighted_recall`
- `val/weighted_precision`

## 注意事项

1. **离线模式限制**：只读取最终的summary文件，无法追踪训练过程中的最佳epoch
2. **在线模式优势**：可以获取完整训练历史，找到训练过程中任何时刻的最佳指标
3. 如果需要分析训练过程中的最佳epoch（而不是最终epoch），请使用在线模式

## 命令行参数

```
--wandb-dir    wandb目录路径（默认：./wandb）
--metric       要追踪的指标名称（默认：val/macro_auprc）
--run-id       指定要分析的run ID
--all          显示所有runs的摘要
--online       使用wandb在线API
--project      W&B项目名称（在线模式必需）
--entity       W&B用户名/组织名（在线模式可选）
```

## 依赖

- Python 3.6+
- wandb（在线模式需要）
- PyYAML（读取配置文件，可选）
