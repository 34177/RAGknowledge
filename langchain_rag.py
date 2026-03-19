"""
LangChain-based RAG System
重构版本，使用LangChain实现
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config


class LangChainRAG:
    """基于LangChain的RAG系统"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

        # 嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        # 向量存储
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.rag_chain = None
        self.llm = None

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """加载PDF文档"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    def load_pdfs(self, pdf_dir: str) -> List[Document]:
        """加载目录下的所有PDF"""
        pdf_path = Path(pdf_dir)
        all_docs = []

        for pdf_file in pdf_path.glob("*.pdf"):
            try:
                docs = self.load_pdf(str(pdf_file))
                all_docs.extend(docs)
                print(f"Loaded: {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")

        return all_docs

    def build_knowledge_base(self, pdf_dir: str, save_path: Optional[str] = None):
        """构建知识库"""
        # 加载文档
        documents = self.load_pdfs(pdf_dir)

        # 文本分割
        chunks = self.text_splitter.split_documents(documents)
        print(f"Total chunks: {len(chunks)}")

        # 创建向量存储
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # 保存向量库
        if save_path:
            self.vectorstore.save_local(save_path)
            print(f"Vector store saved to: {save_path}")

        # 设置检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        return self.vectorstore

    def load_knowledge_base(self, load_path: str):
        """加载已有知识库"""
        self.vectorstore = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        return self.vectorstore

    def set_llm(self, provider: str = "qwen", **kwargs):
        """设置LLM"""
        if provider == "qwen":
            from langchain_community.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(
                model=kwargs.get("model", "qwen-plus"),
                api_key=kwargs.get("api_key", Config.QWEN_API_KEY),
                base_url=kwargs.get("base_url", Config.QWEN_BASE_URL),
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1024)
            )
        elif provider == "claude":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=kwargs.get("model", "claude-3-5-sonnet-20241022"),
                anthropic_api_key=kwargs.get("api_key", os.environ.get("ANTHROPIC_API_KEY")),
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1024)
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # 构建RAG链
        self._build_chain()

    def _build_chain(self):
        """构建RAG链"""
        # 定义提示模板
        template = """你是一个专业的知识问答助手。请根据以下参考文档回答用户的问题。

参考文档：
{context}

用户问题：{question}

请给出准确、详细的回答。如果参考文档中没有相关信息，请明确说明。"""

        prompt = ChatPromptTemplate.from_template(template)

        # 构建链
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def answer(self, question: str, return_context: bool = False):
        """回答问题"""
        if not self.rag_chain:
            raise ValueError("LLM not set. Call set_llm() first.")

        # 获取检索结果
        docs = self.retriever.invoke(question)

        # 过滤低相似度
        if self.similarity_threshold > 0:
            filtered_docs = []
            for doc in docs:
                # 计算相似度（简化处理，使用距离转换）
                # 注意：LangChain的similarity_search_with_score返回距离，需要转换
                score = 1 / (1 + getattr(doc, 'metadata', {}).get('score', 0))
                if score >= self.similarity_threshold:
                    filtered_docs.append(doc)

            if not filtered_docs:
                return {
                    "answer": "抱歉，根据相似度阈值过滤，未找到足够相关的参考文档，无法回答您的问题。",
                    "context": [],
                    "scores": []
                }

            docs = filtered_docs

        # 生成答案
        answer = self.rag_chain.invoke(question)

        if return_context:
            return {
                "answer": answer,
                "context": [doc.page_content for doc in docs],
                "sources": [doc.metadata.get("source", "unknown") for doc in docs]
            }

        return answer

    def similarity_search_with_scores(self, query: str, k: int = None):
        """带分数的相似度搜索"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        k = k or self.top_k
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "document": doc,
                "score": score,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown")
            }
            for doc, score in results
        ]


# 便捷函数
def create_rag(
    pdf_dir: str,
    vector_store_path: str = None,
    llm_provider: str = "qwen",
    **kwargs
) -> LangChainRAG:
    """创建RAG系统"""
    rag = LangChainRAG(
        chunk_size=kwargs.get("chunk_size", 512),
        chunk_overlap=kwargs.get("chunk_overlap", 50),
        top_k=kwargs.get("top_k", 5),
        similarity_threshold=kwargs.get("similarity_threshold", 0.7)
    )

    # 构建或加载知识库
    if vector_store_path and Path(vector_store_path).exists():
        print(f"Loading existing vector store from: {vector_store_path}")
        rag.load_knowledge_base(vector_store_path)
    else:
        print(f"Building knowledge base from: {pdf_dir}")
        rag.build_knowledge_base(pdf_dir, vector_store_path)

    # 设置LLM
    rag.set_llm(llm_provider, **kwargs)

    return rag
