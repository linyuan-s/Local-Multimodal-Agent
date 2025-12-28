# Local Multimodal AI Agent (本地多模态 AI 智能助手)

基于本地运行的多模态 AI 助手，支持文献语义搜索、自动分类整理以及以文搜图功能。

## 核心功能

*   **📄 智能文献管理**
    *   **语义搜索**: 使用自然语言（如“Transformer 的核心架构”）搜索本地论文库，支持返回具体页码和文本片段。
    *   **自动分类**: 自动根据论文摘要和预定义的 Topics（如 SGG, Hypergraph, RL）进行分类整理。本系统**特别适配** Scene Graph Generation (SGG), Hypergraph, Reinforcement Learning (RL) 等领域的文献管理。
    *   **全文索引**: 利用 `PyMuPDF` 高效解析 PDF，结合向量数据库实现精准检索。
*   **🖼️ 智能图像管理**
    *   **以文搜图**: 输入描述（如“海边的日落”），利用 CLIP 模型快速找到本地图片。
*   **🔒 本地隐私**
    *   所有数据和模型均在本地运行 (ChromaDB + SentenceTransformers)，无需上传数据，保障隐私。

## 快速开始

### 1. 环境准备

本项目使用 [uv](https://github.com/astral-sh/uv) 进行极速环境管理。

```bash
# 1. 克隆仓库
git clone https://github.com/linyuan-s/Local-Multimodal-Agent.git
cd Local-Multimodal-Agent

# 2. 创建并激活环境 (推荐使用 Conda 隔离 uv，或者直接使用 uv)
# 如果您已经按照教程创建了 multi_env：
conda activate multi_env

# 3. 初始化项目并同步依赖
uv sync
```

### 2. 使用方法

#### 命令行接口 (CLI)

项目提供 `main.py` 作为统一入口。

**添加并分类论文**
```bash
python main.py add-paper "path/to/paper.pdf" --topics "SGG,Hypergraph,RL"
```
*   系统出会自动提取文本，计算 Embedding，归类到对应 Topic 文件夹，并将索引存入数据库。

**批量导入 (Ingest)**
```bash
python main.py ingest "path/to/folder" --topics "SGG,Hypergraph,RL"
```
*   自动扫描文件夹下的所有 PDF 和图片进行入库。

**搜索论文**
```bash
python main.py search-paper "Transformer 的注意力机制是什么？"
```
*   返回最相关的论文片段及其所在的页码。

**索引图像**
```bash
python main.py index-image "path/to/image_folder_or_file"
```

**以文搜图**
```bash
python main.py search-image "一只狗"
```

## 技术栈

*   **Language**: Python 3.10+
*   **Vector DB**: ChromaDB (Persistent)
*   **Embedding**:
    *   Text: `all-MiniLM-L6-v2` (Sentence-Transformers)
    *   Image: `CLIP` (OpenAI/HuggingFace)
*   **PDF Parsing**: PyMuPDF (fitz)
*   **CLI**: Typer

## 项目结构
```
.
├── main.py                 # CLI 入口
├── src/
│   ├── core/               # 数据库与模型核心逻辑
│   └── services/           # 业务逻辑 (论文/图像处理)
├── chroma_db/              # 本地向量数据库存储
└── pyproject.toml          # 项目依赖配置
```
