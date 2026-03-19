"""
LangChain RAG System - Main Entry
使用LangChain实现的RAG知识问答系统
"""

import os
from pathlib import Path
from langchain_rag import LangChainRAG
from config import Config


def main():
    print("=" * 50)
    print("LangChain RAG 知识问答系统")
    print("=" * 50)

    # PDF目录
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
        print(f"\n加载已有知识库: {vector_store_path}")
        rag.load_knowledge_base(vector_store_path)
    else:
        print(f"\n构建知识库 from: {pdf_dir}")
        # 创建目录
        os.makedirs("data/vectors", exist_ok=True)
        rag.build_knowledge_base(pdf_dir, vector_store_path)

    # 设置LLM
    print(f"\n设置LLM: {Config.LLM_PROVIDER}")
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

    print("\n" + "=" * 50)
    print("系统就绪！输入问题进行问答（输入 q 退出）")
    print("=" * 50)

    # 问答循环
    while True:
        question = input("\n问题: ").strip()

        if question.lower() in ['q', 'quit', 'exit']:
            print("再见！")
            break

        if not question:
            continue

        try:
            result = rag.answer(question, return_context=True)
            print(f"\n回答:\n{result['answer']}")

            if result.get('context'):
                print(f"\n参考来源:")
                for i, (ctx, src) in enumerate(zip(result['context'][:3], result.get('sources', [])[:3]), 1):
                    print(f"  [{i}] {src}")
                    print(f"      {ctx[:200]}...")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
