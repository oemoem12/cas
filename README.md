# caS - catch AI models on your PC or Server

一个本地 AI 模型部署工具，灵感来自 Ollama。

## 特性

- **多模型源支持**: HuggingFace、hf-mirror（国内镜像）、魔塔社区（ModelScope）
- **GGUF 量化模型**: 支持下载指定量化版本（Q4_K_M、Q8_0 等），节省 70%+ 磁盘空间
- **多种推理模式**: 单次推理、多轮对话、REST API 服务
- **自动配置**: 自动从 GGUF 文件生成 config.json 和下载 tokenizer

## 安装

### 方法一：使用 install.sh

```bash
git clone https://github.com/YOUR_USERNAME/cas.git
cd cas
bash install.sh
```

### 方法二：使用 deb 包

```bash
sudo dpkg -i cas_0.1.0_all.deb
```

### 方法三：使用 pip

```bash
pip install git+https://github.com/YOUR_USERNAME/cas.git
```

## 快速开始

### 下载模型

```bash
# 从 HuggingFace 下载
cas pull Qwen/Qwen2.5-0.5B --source huggingface

# 从 hf-mirror 国内镜像下载 GGUF 量化模型
cas pull bartowski/Qwen2.5-0.5B-Instruct-GGUF --gguf --quant Q4_K_M --source hf-mirror

# 从魔塔社区下载
cas pull Qwen/Qwen2.5-0.5B --source modelscope
```

### 推理

```bash
# 单次推理
cas run bartowski/Qwen2.5-0.5B-Instruct-GGUF "What is AI?" --max-tokens 100

# 指定量化版本
cas run bartowski/Qwen2.5-0.5B-Instruct-GGUF "Hello" --quant Q4_K_M
```

### 对话模式

```bash
cas chat bartowski/Qwen2.5-0.5B-Instruct-GGUF
cas chat bartowski/Qwen2.5-0.5B-Instruct-GGUF --system "你是一个编程助手"
```

### 启动 API 服务

```bash
cas serve --port 8000

# 测试 API
curl http://localhost:8000/api/models
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B","prompt":"Hello","max_tokens":100}'
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/pull` | POST | 下载模型 |
| `/api/models` | GET | 列出本地模型 |
| `/api/generate` | POST | 生成文本 |

## 支持的 GGUF 量化格式

Q8_0, Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q4_0, Q3_K_M, Q3_K_S, Q2_K, IQ2_M, IQ3_M, IQ4_XS, f16

## 许可证

GPL-3.0
