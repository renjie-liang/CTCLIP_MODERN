# Dataset Split Parameter 诊断报告

## 问题描述
用户报告：运行 `python scripts_important/build_npz_from_hf.py --split valid` 时下载的是 train dataset

## 代码分析结果

### ✅ 代码逻辑检查

经过详细检查，**当前代码逻辑完全正确**：

1. **参数解析**（第 731-737 行）
   ```python
   parser.add_argument(
       '--split',
       type=str,
       required=True,
       choices=['train', 'valid'],
       help='Dataset split to process'
   )
   ```
   ✅ 参数定义正确

2. **配置使用**（第 748 行 & 766 行）
   ```python
   output_dir = Path(SPLIT_CONFIGS[args.split]['output_dir'])
   local_source_dir = Path(LOCAL_SOURCE_DIRS[args.split])
   ```
   ✅ 使用 `args.split` 正确索引配置

3. **文件过滤**（第 645-656 行）
   ```python
   def list_hf_files(split: str) -> List[str]:
       path_pattern = SPLIT_CONFIGS[split]['hf_path_pattern']
       split_files = [
           f for f in all_files
           if f.startswith(path_pattern) and f.endswith('.nii.gz')
       ]
   ```
   ✅ 使用正确的 pattern 过滤文件

### ✅ 配置文件检查

`config_npz_conversion.py` 中的配置：

```python
SPLIT_CONFIGS = {
    'train': {
        'hf_path_pattern': 'dataset/train_fixed',  # 只匹配 train 文件
        'output_dir': '/orange/.../train_npz'
    },
    'valid': {
        'hf_path_pattern': 'dataset/valid_fixed',  # 只匹配 valid 文件
        'output_dir': '/orange/.../valid_npz'
    }
}

LOCAL_SOURCE_DIRS = {
    'train': '/orange/.../train_fixed',
    'valid': '/orange/.../valid_fixed'
}
```

✅ 配置完全正确

### ✅ 功能验证

运行诊断脚本验证：

```bash
$ python3 diagnose_split_issue.py --split valid
```

结果显示：
- ✓ `args.split` 正确解析为 'valid'
- ✓ 使用的 hf_path_pattern 为 'dataset/valid_fixed'
- ✓ 只匹配了 valid 文件，**没有匹配任何 train 文件**

## 可能的问题原因

既然代码是正确的，那么可能的原因是：

### 1. 误读输出信息
运行时可能看到类似信息：
```
Scanning local directory: /orange/.../valid_fixed
Found 498 local nii.gz files
```

如果目录路径显示有问题，可能被误认为在扫描 train 目录。

### 2. 之前运行过错误的命令
可能实际上运行的是：
```bash
python scripts_important/build_npz_from_hf.py --split train  # 注意是 train!
```

### 3. 配置文件被临时修改
在运行时，配置文件可能被手动修改过，导致：
- `SPLIT_CONFIGS['valid']['hf_path_pattern']` 指向了 'dataset/train_fixed'
- 或者 `LOCAL_SOURCE_DIRS['valid']` 指向了 train 目录

## 如何验证问题

### 方法 1：运行诊断脚本

```bash
# 在项目根目录运行
python3 diagnose_split_issue.py --split valid
```

检查输出中的：
- HF path pattern 是否为 'dataset/valid_fixed'
- 是否只匹配了 valid 文件

### 方法 2：运行测试并检查输出

```bash
python3 test_split_param.py
```

查看两个 split 的配置对比

### 方法 3：实际运行脚本（处理 1 个文件）

```bash
python scripts_important/build_npz_from_hf.py --split valid --max-files 1
```

仔细检查输出的：
- "Split: valid" 行
- "Listing files from ... (split=valid)" 行
- "Found X valid files on HuggingFace" 行
- 实际处理的文件名（应该包含 "valid"）

## 结论

**当前代码没有问题！** `--split valid` 确实会正确地：
1. 只列出 HuggingFace 上 `dataset/valid_fixed/` 路径下的文件
2. 只扫描本地 `/orange/.../valid_fixed/` 目录
3. 只下载和处理 valid 数据集的文件
4. 输出到 `/orange/.../valid_npz/` 目录

如果仍然遇到问题，请提供：
1. 完整的运行命令
2. 脚本的输出日志（前 50 行）
3. 实际生成的文件路径示例

这样可以帮助定位实际问题所在。
