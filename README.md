# 本地多模态 AI 智能助手 (Local Multimodal AI Agent)

>  **基于大模型的本地化文献整理、语义检索与以文搜图系统**

本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术（Sentence-BERT, CLIP），实现对内容的**语义搜索**、**自动分类**以及**以文搜图**。

本项目包含 **命令行 (CLI)** 和 **可视化界面 (Web UI)** 两种交互方式，完全本地运行，保护隐私且无需联网。

---

##  核心功能 (Core Features)

### 1. 智能自动分类 (Smart Organization)
*   **功能**: 一键扫描混乱的文件夹，自动识别 PDF 论文的主题（如 "SGG", "RL", "Hypergraph"）。
*   **技术**: 基于语义嵌入 (Embedding) 的 Zero-shot 分类，将论文移动到对应的子文件夹中。
*   **解决痛点**: 告别手动整理 PDF 的繁琐。

### 2. 深度语义文献检索 (Semantic Paper Search)
*   **功能**: 支持自然语言提问（如“Hypergraph 的核心思想是什么？”）。系统返回最相关的论文片段、页码，并支持**高亮显示**。
*   **技术**: Sentence-BERT 文本向量化 + ChromaDB 向量检索。
*   **解决痛点**: 即使忘记文件名，也能通过内容找回文献。

### 3. 图像搜索 (Neural Image Search)
*   **功能**: “以文搜图”，输入自然语言描述（如“A cute dog on the grass”），即可找到库中最匹配的图片。
*   **技术**: OpenAI CLIP 多模态模型。
*   **解决痛点**: 能够搜索没有标签和文件名的图片素材。

---

## 技术栈 (Tech Stack)

*   **语言**: Python 3.10+
*   **前端**: `Streamlit` (Web 可视化界面)
*   **核心模型**:
    *   文本嵌入: `all-MiniLM-L6-v2` (Sentence-Transformers)
    *   多模态嵌入: `openai/clip-vit-base-patch32` (HuggingFace Transformers)
*   **向量数据库**: `ChromaDB` (本地持久化存储)
*   **PDF 处理**: `PyMuPDF` (fitz)
*   **依赖管理**: `uv` + `Conda`

---

## 快速开始 (Quick Start)

### 1. 环境准备
确保已安装 Python 3.10+ (推荐使用 Conda)。

```bash
# 创建并激活环境
conda create -n multi_env python=3.10
conda activate multi_env

# 安装依赖 (推荐使用 uv 加速)
pip install uv
uv pip install -r requirements.txt
# 或者直接安装核心库:
# uv pip install chromium transformers sentence-transformers pymupdf typer streamlit watchdog
```

### 2. 启动可视化界面 (推荐)
这是最直观的使用方式。

```bash
streamlit run app.py
```
启动后浏览器会自动打开，您可以：
1.  输入文件夹路径进行 **自动整理**。
2.  输入问题进行 **文献检索**。
3.  输入描述进行 **以文搜图**。

### 3. 命令行模式 (CLI Usage)
如果您喜欢终端操作，也可以使用 `main.py`。

**自动导入与整理**:
```bash
python main.py ingest "D:\您的文件夹路径" --topics "SGG,Hypergraph,RL"
```

**搜论文**:
```bash
python main.py search-paper "What is Scene Graph Generation?"
```

**搜图片**:
```bash
python main.py search-image "A dog"
```

---


## 项目结构
```
.
├── app.py                  # Streamlit 前端主程序
├── main.py                 # CLI 命令行入口
├── src/
│   ├── core/               # 核心模块 (数据库, 模型加载, PDF处理)
│   └── services/           # 业务逻辑 (论文服务, 图像服务)
├── README.md               # 项目文档
└── .gitignore              # Git 忽略配置
```

---
*Created by 宋奕霖.*
