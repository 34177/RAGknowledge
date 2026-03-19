"""
LangChain RAG System - Streamlit Visualization
基于LangChain的RAG系统Web界面
"""

import streamlit as st
from pathlib import Path
from langchain_rag import LangChainRAG
from config import Config


@st.cache_resource
def get_rag_system():
    """初始化RAG系统（缓存）"""
    pdf_dir = "book"
    vector_store_path = "data/vectors/faiss_index"

    # 创建RAG系统
    rag = LangChainRAG(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        top_k=Config.TOP_K,
        similarity_threshold=Config.SIMILARITY_THRESHOLD
    )

    # 检查是否存在已有向量库
    if Path(vector_store_path).exists():
        rag.load_knowledge_base(vector_store_path)
    else:
        st.warning("知识库未构建，请先运行 main_langchain.py 构建知识库")
        return None

    # 设置LLM
    if Config.LLM_PROVIDER == "qwen":
        rag.set_llm(
            provider="qwen",
            model=Config.QWEN_MODEL,
            api_key=Config.QWEN_API_KEY,
            base_url=Config.QWEN_BASE_URL
        )
    else:
        rag.set_llm(
            provider="claude",
            model=Config.CLAUDE_MODEL
        )

    return rag


def main():
    st.set_page_config(
        page_title="LangChain RAG 知识问答系统",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 LangChain RAG 知识问答系统")
    st.markdown("基于LangChain + FAISS + Qwen的专业知识问答")

    # 侧边栏配置
    st.sidebar.header("配置")

    top_k = st.sidebar.slider("Top-K 检索数量", 1, 10, Config.TOP_K)
    threshold = st.sidebar.slider("相似度阈值", 0.0, 1.0, Config.SIMILARITY_THRESHOLD, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**技术栈**")
    st.sidebar.markdown("- LangChain")
    st.sidebar.markdown("- FAISS")
    st.sidebar.markdown("- Sentence-Transformers")
    st.sidebar.markdown(f"- {Config.LLM_PROVIDER.upper()}")

    # 初始化RAG系统
    rag = get_rag_system()

    if rag is None:
        st.info("请先构建知识库后刷新页面")
        return

    # 更新配置
    rag.top_k = top_k
    rag.similarity_threshold = threshold

    # 问答界面
    question = st.text_input("请输入您的问题:", placeholder="例如：什么是操作系统？")

    if question:
        with st.spinner("检索中..."):
            # 获取检索结果（带分数）
            search_results = rag.similarity_search_with_scores(question, k=top_k)

            # 显示检索结果
            st.subheader("📚 检索结果")

            for i, result in enumerate(search_results, 1):
                score = result['score']
                # 转换距离为相似度（FAISS返回的是距离）
                similarity = 1 / (1 + score)

                with st.expander(f"结果 {i} - 相似度: {similarity:.3f}"):
                    st.markdown(f"**来源**: {result['source']}")
                    st.markdown(f"**内容**:")
                    st.text(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])

            # 生成答案
            if similarity >= threshold:
                with st.spinner("生成答案中..."):
                    answer = rag.answer(question)

                st.subheader("💡 回答")
                st.markdown(answer)
            else:
                st.warning("⚠️ 相似度低于阈值，无法生成可靠答案")

    # 显示统计信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("**知识库统计**")
    if rag.vectorstore:
        st.sidebar.text(f"向量维度: {Config.EMBEDDING_DIM}")
        st.sidebar.text(f"文档数量: {rag.vectorstore.index.ntotal}")


if __name__ == "__main__":
    main()
