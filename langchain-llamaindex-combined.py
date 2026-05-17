#!/usr/bin/env python
# coding: utf-8
"""
LangChain + LlamaIndex 组合版：多文件智能问答 + 报告生成 + 邮件发送

架构思路：
  - LlamaIndex 负责「数据层」：文档加载、分块、向量索引、语义检索
  - LangChain  负责「编排层」：LCEL 问答链、Agent 工具调用（生成报告、发送邮件）
  - 两者通过 Retriever 接口对接：LlamaIndex 的检索结果转为 LangChain Document
"""

import os
import json
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LCDocument
from langchain_core.tools import tool

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# ============================================================
# 第一部分：LlamaIndex 负责数据层（索引 + 检索）
# ============================================================

def setup_llamaindex():
    """配置 LlamaIndex 的 LLM 和 Embedding（全局设置）"""
    llm = DashScope(
        model="deepseek-v3.2",
        api_key=DASHSCOPE_API_KEY,
        temperature=0.1,
        top_p=0.8,
    )
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def build_llamaindex_index(file_dir: str = './docs', persist_dir: str = './combined_storage'):
    """用 LlamaIndex 加载文档并构建向量索引（自动分块 + 向量化）"""

    if os.path.exists(persist_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("[LlamaIndex] 从本地存储加载索引成功")
            return index
        except Exception as e:
            print(f"[LlamaIndex] 加载索引失败: {e}，将重新创建")

    if not os.path.exists(file_dir):
        print(f"[LlamaIndex] 文档目录 {file_dir} 不存在")
        return None

    documents = SimpleDirectoryReader(file_dir).load_data()
    if not documents:
        print("[LlamaIndex] 没有找到任何文档")
        return None

    print(f"[LlamaIndex] 加载了 {len(documents)} 个文档，正在构建索引...")
    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=persist_dir)
    print(f"[LlamaIndex] 索引已保存到 {persist_dir}")
    return index


def llamaindex_retrieve(index, query: str, top_k: int = 5):
    """用 LlamaIndex 检索，将结果转为 LangChain 的 Document 格式"""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    lc_docs = []
    for node in nodes:
        lc_docs.append(LCDocument(
            page_content=node.text,
            metadata={
                "score": round(node.score, 4) if node.score else None,
                "file_name": node.metadata.get("file_name", "未知"),
            }
        ))
    return lc_docs


# ============================================================
# 第二部分：LangChain 负责编排层（问答链 + 工具）
# ============================================================

def create_qa_chain(llm):
    """用 LangChain LCEL 构建问答链"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的保险产品顾问。
根据以下检索到的文档内容回答用户的问题。
如果文档中没有相关信息，请如实说明。请用中文回复。

检索到的文档内容:
{context}"""),
        ("human", "{question}")
    ])
    return qa_prompt | llm | StrOutputParser()


def create_report_chain(llm):
    """用 LangChain LCEL 构建报告生成链"""
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的保险产品分析师。
根据以下问答记录，生成一份结构化的分析报告。
报告需包含：概述、核心要点、适用场景、注意事项。
请用中文撰写，语言简洁专业。

问答记录:
问题: {question}
回答: {answer}

相关文档来源: {sources}"""),
        ("human", "请生成分析报告")
    ])
    return report_prompt | llm | StrOutputParser()


@tool
def send_email(recipient: str, subject: str, content: str) -> str:
    """发送邮件工具。将报告通过邮件发送给指定收件人。

    参数:
        recipient: 收件人邮箱地址
        subject: 邮件主题
        content: 邮件正文内容
    返回:
        发送结果
    """
    # 模拟邮件发送（实际项目中对接 SMTP 或邮件 API）
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- 模拟发送邮件 ---")
    print(f"  收件人: {recipient}")
    print(f"  主题:   {subject}")
    print(f"  时间:   {timestamp}")
    print(f"  正文长度: {len(content)} 字")
    print(f"--- 邮件发送成功 ---\n")
    return f"邮件已于 {timestamp} 成功发送至 {recipient}，主题: {subject}"


@tool
def save_report(filename: str, content: str) -> str:
    """保存报告到本地文件。

    参数:
        filename: 文件名（不含路径）
        content: 报告内容
    返回:
        保存结果
    """
    output_dir = "./reports"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[保存报告] 已保存到 {filepath}")
    return f"报告已保存到 {filepath}"


# ============================================================
# 第三部分：主流程 —— 串联 LlamaIndex 检索 + LangChain 编排
# ============================================================

def main():
    """
    完整流程:
    1. LlamaIndex 建索引 + 检索
    2. LangChain 问答链生成回答
    3. LangChain 报告链生成报告
    4. LangChain 工具调用保存报告 + 发送邮件
    """

    # --- 步骤 1：初始化 LlamaIndex（数据层） ---
    print("=" * 60)
    print("步骤 1：初始化 LlamaIndex 索引")
    print("=" * 60)
    setup_llamaindex()
    index = build_llamaindex_index()
    if index is None:
        print("无法创建索引，程序退出")
        return

    # --- 步骤 2：初始化 LangChain（编排层） ---
    print("\n" + "=" * 60)
    print("步骤 2：初始化 LangChain 编排链")
    print("=" * 60)
    langchain_llm = ChatTongyi(
        model_name="deepseek-v3.2",
        dashscope_api_key=DASHSCOPE_API_KEY
    )
    qa_chain = create_qa_chain(langchain_llm)
    report_chain = create_report_chain(langchain_llm)

    # --- 步骤 3：LlamaIndex 检索 → LangChain 问答 ---
    query = "雇主责任险的保障范围和理赔流程是什么？"
    print(f"\n{'=' * 60}")
    print(f"步骤 3：执行检索与问答")
    print(f"{'=' * 60}")
    print(f"用户查询: {query}\n")

    # LlamaIndex 检索，结果转为 LangChain Document
    lc_docs = llamaindex_retrieve(index, query, top_k=5)

    print("===== 召回的文档内容（LlamaIndex 检索） =====")
    for i, doc in enumerate(lc_docs):
        preview = doc.page_content[:150]
        print(f"\n片段 {i+1} [来源: {doc.metadata['file_name']}, 相似度: {doc.metadata['score']}]")
        print(f"  {preview}...")
    print("=" * 50 + "\n")

    # 拼接上下文，交给 LangChain 问答链
    context = "\n\n".join(
        f"[来源: {doc.metadata['file_name']}]\n{doc.page_content}"
        for doc in lc_docs
    )
    answer = qa_chain.invoke({"context": context, "question": query})

    print("===== AI 回答（LangChain LCEL 链） =====")
    print(answer)
    print("=" * 50 + "\n")

    # --- 步骤 4：LangChain 报告链生成报告 ---
    print(f"{'=' * 60}")
    print(f"步骤 4：生成分析报告")
    print(f"{'=' * 60}")

    sources = ", ".join(set(doc.metadata['file_name'] for doc in lc_docs))
    report = report_chain.invoke({
        "question": query,
        "answer": answer,
        "sources": sources,
    })

    print("===== 生成的分析报告 =====")
    print(report)
    print("=" * 50 + "\n")

    # --- 步骤 5：调用工具保存报告 + 发送邮件 ---
    print(f"{'=' * 60}")
    print(f"步骤 5：保存报告 & 发送邮件（LangChain 工具调用）")
    print(f"{'=' * 60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"雇主责任险分析报告_{timestamp}.txt"

    save_result = save_report.invoke({
        "filename": filename,
        "content": report,
    })
    print(save_result)

    email_result = send_email.invoke({
        "recipient": "manager@example.com",
        "subject": f"保险产品分析报告 - 雇主责任险",
        "content": report,
    })
    print(email_result)

    print(f"\n{'=' * 60}")
    print("全部流程完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
