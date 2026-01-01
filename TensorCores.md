由于你使用的是 **Ubuntu 20.04** 并且配备了 **A100** 显卡，安装 **DCGM (Data Center GPU Manager)** 是监控 Tensor Core 和 GPU 状态的最佳选择。

以下是标准安装步骤：

### 1. 添加 NVIDIA 官方存储库
首先，你需要配置 NVIDIA 的软件源，以便通过 `apt` 安装 DCGM。

```bash
# 下载并添加 NVIDIA GPG 密钥
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 添加存储库（针对 Ubuntu 20.04）
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

### 2. 更新索引并安装 DCGM
```bash
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
```

### 3. 启动并启用 DCGM 服务
DCGM 需要一个后台守护进程（`nv-hostengine`）来收集数据。
```bash
sudo systemctl enable nvidia-dcgm
sudo systemctl start nvidia-dcgm

# 检查服务状态，确保是 active (running)
sudo systemctl status nvidia-dcgm
```

### 4. 验证安装
使用 `dcgmi` 命令列出系统中识别到的 GPU：
```bash
dcgmi discovery -l
```
如果你看到了 A100 显卡的信息，说明安装成功。

---

### 如何使用 DCGM 监控 Tensor Core？

安装完成后，你可以实时查看 Tensor Core 的利用率。

#### 实时监控命令：
针对 A100，我们主要关注 **FP16** 和 **BF16** 的 Tensor Core 活动比例（指标 ID 分别为 1004 和 1006）：

```bash
# 每秒刷新一次，显示所有 GPU 的 Tensor Core 活动比
# 指标 1004: Tensor Core FP16 活动占比
# 指标 1006: Tensor Core BF16 活动占比
# 指标 1002: 普通 CUDA Core (FP32) 活动占比
dcgmi dmon -e 1004,1006,1002 -i 0
```

**字段说明：**
*   **ID**: GPU 编号。
*   **1004 (FB16_ACTIVE)**: 如果你在用 `float16` 训练，看这个值。如果是 0.20，意味着当前 Tensor Core 达到了理论峰值吞吐量的 20%。
*   **1006 (BF16_ACTIVE)**: 如果你在用 `bfloat16` 训练，看这个值。
*   **1002 (FP32_ACTIVE)**: 普通 CUDA Core 的利用率。

---

### 常见问题排查：
1.  **权限问题**：如果执行 `dcgmi` 报错，请尝试在命令前加 `sudo`。
2.  **找不到服务**：如果提示 `nvidia-dcgm` 不存在，请确保在第一步添加存储库时没有报错，并重新执行 `apt-get update`。
3.  **显示均为 0**：Tensor Core 只有在模型运行**矩阵乘法（GEMM）**且数据类型为 **FP16/BF16** 时才会瞬间飙升。如果你只运行一个简单的 Python 脚本，数据采集可能捕捉不到那一瞬间的峰值。建议在模型训练过程中长时间观察。

如果你需要将这些数据导出到可视化界面（如 Prometheus/Grafana），可以进一步安装 `dcgm-exporter`。