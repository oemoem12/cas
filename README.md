# caS - catch AI models on your PC or Server

caS 是一个**本地 AI 模型部署工具**，灵感来自 Ollama。它让你在自己的电脑或服务器上轻松下载、管理和运行开源大语言模型，无需依赖任何云服务。

**核心功能：**
- 从多个模型源下载模型（HuggingFace、hf-mirror 国内镜像、魔塔社区）
- 支持 GGUF 量化格式，大幅降低内存和磁盘占用
- 提供命令行交互（单次推理 / 多轮对话）和 REST API 两种方式
- 自动处理 GGUF 模型的 config.json 和 tokenizer

---

## 安装

### 方法一：使用 install.sh（推荐）

```bash
git clone https://github.com/oemoem12/cas.git
cd cas
bash install.sh
```

### 方法二：使用 deb 包（Debian/Ubuntu）

```bash
sudo dpkg -i cas_0.1.0_all.deb
```

### 方法三：使用 pip

```bash
pip install git+https://github.com/oemoem12/cas.git
```

---

## 使用方法

### 1. 下载模型

```bash
cas pull <模型ID> [选项]
```

| 选项 | 说明 |
|------|------|
| `--source huggingface` | 从 HuggingFace 官方源下载（默认） |
| `--source hf-mirror` | 从 hf-mirror.com 国内镜像下载 |
| `--source modelscope` | 从魔塔社区下载 |
| `--gguf` | 仅下载 GGUF 量化格式（节省 70%+ 空间） |
| `--quant Q4_K_M` | 指定量化版本，只下载该文件 |

```bash
# 示例：下载 GGUF 量化模型（推荐，内存占用小）
cas pull bartowski/Qwen2.5-0.5B-Instruct-GGUF --gguf --quant Q4_K_M --source hf-mirror

# 示例：下载完整 safetensors 模型（需要更多内存）
cas pull Qwen/Qwen2.5-0.5B --source hf-mirror
```

### 2. 单次推理

```bash
cas run <模型ID> "你的问题" [选项]
```

| 选项 | 说明 |
|------|------|
| `--max-tokens 100` | 最大生成 token 数（默认 100） |
| `--temperature 0.7` | 温度值，越高越随机（默认 0.7） |
| `--quant Q4_K_M` | 指定使用哪个量化版本 |

```bash
cas run bartowski/Qwen2.5-0.5B-Instruct-GGUF "什么是机器学习？" --max-tokens 200
```

### 3. 多轮对话

```bash
cas chat <模型ID> [选项]
```

| 选项 | 说明 |
|------|------|
| `--system "你是助手"` | 设置系统提示词 |
| `--max-tokens 512` | 最大生成 token 数（默认 512） |
| `--temperature 0.7` | 温度值 |
| `--quant Q4_K_M` | 指定量化版本 |

```bash
cas chat bartowski/Qwen2.5-0.5B-Instruct-GGUF
cas chat bartowski/Qwen2.5-0.5B-Instruct-GGUF --system "你是一个编程助手"
```

对话中输入 `clear` 清空历史，输入 `quit` 或 `exit` 退出。

### 4. 管理模型

```bash
cas list                  # 列出已下载的模型
cas list --verbose        # 显示详细信息（来源、类型、大小、路径）
cas rm <模型ID>           # 删除指定模型
cas rm-all                # 删除所有模型
```

### 5. 启动 API 服务

```bash
cas serve --port 8000
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/pull` | POST | 下载模型 |
| `/api/models` | GET | 列出本地模型 |
| `/api/generate` | POST | 生成文本 |

```bash
# 测试 API
curl http://localhost:8000/api/models

curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B","prompt":"Hello","max_tokens":100}'
```

---

## 详细信息

### GGUF 模型从哪来？

GGUF 是由 [llama.cpp](https://github.com/ggerganov/llama.cpp) 项目定义的量化模型格式。caS 从 HuggingFace 上的第三方发布者获取 GGUF 文件，推荐以下发布者：

| 发布者 | 说明 |
|--------|------|
| `bartowski` | 覆盖面广，量化格式齐全 |
| `MaziyarPanahi` | 常用模型整合 |
| `TheBloke` | 经典量化版本（部分模型） |

搜索方式：在 [HuggingFace](https://huggingface.co) 搜索 `模型名 + GGUF` 即可找到量化版本。

### 支持的操作系统

| 系统 | 支持情况 |
|------|----------|
| **Linux** (Debian/Ubuntu) | ✅ 完整支持，提供 deb 包 |
| **Linux** (其他发行版) | ✅ 支持 pip / install.sh 安装 |
| **macOS** | ✅ 支持 pip 安装（需 Apple Silicon 或 Intel） |
| **Windows** | ⚠️ 可通过 WSL2 使用 |

### 硬件要求

| 模型类型 | 最低内存 | 推荐内存 |
|----------|----------|----------|
| GGUF Q4_K_M (0.5B) | ~1 GB | 2 GB |
| GGUF Q4_K_M (2B) | ~1.5 GB | 4 GB |
| GGUF Q4_K_M (7B) | ~4 GB | 8 GB |
| GGUF Q8_0 (7B) | ~8 GB | 16 GB |
| Safetensors (7B) | ~14 GB | 16 GB+ |

> 使用 GGUF 量化模型可显著降低内存需求。`Q4_K_M` 是性价比最高的选择。

### 模型存储位置

所有模型默认存储在 `~/.cas/models/`，可通过修改代码中的 `CACHE_DIR` 自定义路径。

### 支持的 GGUF 量化格式

Q8_0, Q6_K, Q6_K_L, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q4_0, Q3_K_M, Q3_K_S, Q2_K, IQ2_M, IQ3_M, IQ4_XS, f16

不指定 `--quant` 时，caS 按以下优先级自动选择：Q8_0 → Q6_K → Q5_K_M → Q4_K_M → Q4_K_S → Q3_K_M → Q2_K

---

## 许可证

GPL-3.0
