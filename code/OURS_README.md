# OURS 项目说明

---

## 运行环境

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| CUDA | 11.8 |
| PyTorch | 2.1.2+cu118 |
| OS | Ubuntu 22.04 |
| GPU | 4090D × 4 |

> ⚠️ **重要提示**：请勿使用新架构显卡（如 5090）。经多次尝试均未成功，兼容性较差。

---

## 依赖列表

```
opencv-python==4.5.5.62
tqdm
pytorch-msssim
timm
tensorboard
tensorboardx
```

---

## 运行前配置

### 1. 训练 (`train.py`)

| 配置项 | 参数 | 说明 |
|--------|------|------|
| 模型保存路径 | `--save_dir` | `default='./saveMix10/'` |
| 数据集路径 | `--data_dir` | `default='/root/autodl-tmp/OURS/dataset/RESIDE-OUT'` |
| GPU 数量 | `--gpu` | `default='0,1,2,3'` |
| 模型超参数版本 | `--exp` | `default='outdoor'` |

### 2. 测试 (`test_IN.py`)

| 配置项 | 参数 | 说明 |
|--------|------|------|
| 数据集路径 | `--data_dir` | `default='./dataset/'` |
| 训练集路径 | `--dataset` | `default='RESIDE-OUT/test'`（相对于 `data_dir` 的二级路径，无需重复填写一级路径） |
| 测试模型路径 | `--save_dir` | `default='./saveMix10/'` |
| 测试结果保存路径 | `--result_dir` | `default='./results/'` |
