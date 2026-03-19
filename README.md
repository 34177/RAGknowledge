# RAG知识问答系统

基于检索增强生成（RAG）架构的专业知识问答系统，专注于计算机科学领域，支持中文专业书籍的智能化问答。

## 系统架构

### 1. 知识库构建

#### PDF解析层
- **pdfplumber**: 高效提取PDF文本内容
- **Tesseract OCR**: 识别扫描版PDF（图片形式）中的文字
- **Markdown标准化**: 将解析结果转换为统一的Markdown格式

#### 文本分块
- **递归字符切片算法**: 采用分层分块策略保持上下文逻辑连贯性
- **Chunk大小**: 512字符，Overlap: 50字符
- 解决传统随机切片导致的语义中断问题

### 2. 检索层

#### 向量存储
- **FAISS**: 高效的向量相似度搜索库
- **Sentence-Transformers**: all-MiniLM-L6-v2 嵌入模型（384维）

#### 检索机制
- **余弦相似度**: 计算查询与文档片段的语义匹配度
- **Top-K召回**: 返回最相关的K个知识片段
- 支持毫秒级大规模向量检索

### 3. 评估层

#### 相似度阈值过滤
- **阈值**: 0.7（可配置）
- 自动拒绝低相关度检索结果，减少噪声干扰

#### 证据对齐
- 在前端高亮标注原文来源
- 实现可追溯的答案生成，增强可信度

### 4. 大语言模型集成

支持双模型切换：
- **Qwen (千问)**: 阿里云通义千问系列
- **Claude**: Anthropic Claude系列

### 5. 可视化应用

基于Streamlit构建的Web交互界面：
- 问答输入与结果显示
- 检索原文高亮展示
- 相似度分数可视化

## 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/34177/RAGknowledge.git
cd RAGknowledge

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置API密钥

编辑 `config.py`：

```python
# Qwen配置
QWEN_API_KEY = "your-qwen-api-key"

# 或 Claude配置
LLM_PROVIDER = "claude"
```

### 构建知识库

```bash
python main.py
```

### 启动Web界面

```bash
streamlit run visualization/app.py
```

## 项目结构

```
RAGknowledge/
├── knowledge_base/         # 知识库构建模块
│   ├── nougat_parser.py   # PDF解析器
│   ├── text_splitter.py   # 文本分块
│   └── vector_store.py    # 向量存储
├── retrieval/             # 检索模块
│   └── retriever.py       # 向量检索
├── evaluation/            # 评估模块
│   └── threshold_filter.py # 阈值过滤
├── llm/                   # LLM集成
│   ├── qwen_client.py     # 千问客户端
│   └── claude_client.py   # Claude客户端
├── visualization/         # 可视化
│   └── app.py             # Streamlit应用
├── rag_system.py          # RAG系统主逻辑
├── main.py                # 入口脚本
└── config.py              # 配置文件
```

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 向量数据库 | FAISS |
| 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| PDF解析 | pdfplumber + Tesseract OCR |
| 大语言模型 | Qwen / Claude |
| Web框架 | Streamlit |
| Python版本 | 3.8+ |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 512 | 文本块大小 |
| CHUNK_OVERLAP | 50 | 文本块重叠长度 |
| TOP_K | 5 | 检索返回数量 |
| SIMILARITY_THRESHOLD | 0.7 | 相似度阈值 |

## 核心优势

1. **逻辑连贯性**: 递归切片算法保持上下文完整性
2. **高精度检索**: FAISS + 余弦相似度实现语义匹配
3. **可解释性**: 证据对齐机制标注原文来源
4. **灵活性**: 支持多模型切换，适应不同场景
5. **中文优化**: 针对中文PDF提供OCR支持

## 适用场景

- 企业内部知识库问答
- 专业技术文档检索
- 学术资料问答系统
- 产品手册智能客服

## License

MIT License
