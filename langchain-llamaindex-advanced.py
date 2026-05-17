#!/usr/bin/env python
# coding: utf-8
"""
LangChain + LlamaIndex 高级整合版：多文件智能问答Agent v2.0

升级特性：
  - 混合检索：向量检索 + BM25关键词检索 + RRF融合
  - 查询扩展：自动优化检索词，提升召回率
  - 对话记忆：支持多轮对话上下文
  - 质量评估：答案质量自动评分
  - 智能路由：根据问题类型选择处理策略

架构思路：
  - LlamaIndex 负责「数据层」：文档处理、混合检索、查询转换
  - LangChain  负责「编排层」：Agent工具调用、链式编排、记忆管理
  - 两者通过标准化接口深度整合
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LCDocument
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# ============================================================
# 第一部分：LlamaIndex 数据层（高级检索）
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


class HybridRetriever:
    """混合检索器：向量检索 + BM25 + RRF融合"""
    
    def __init__(self, index, top_k: int = 5):
        self.index = index
        self.top_k = top_k
        
        # 向量检索器
        self.vector_retriever = index.as_retriever(
            similarity_top_k=top_k * 2
        )
        
        # BM25关键词检索器
        self.bm25_retriever = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=top_k * 2
        )
        
        # 查询融合检索器（RRF算法）
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            similarity_top_k=top_k,
            mode="reciprocal_rerank",
            use_async=False,
            num_queries=2,
            verbose=False
        )
        
        # 后处理器
        self.postprocessor = SimilarityPostprocessor(
            similarity_cutoff=0.5
        )
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """执行混合检索"""
        nodes = self.fusion_retriever.retrieve(query)
        nodes = self.postprocessor.postprocess_nodes(nodes)
        return nodes[:self.top_k]
    
    def retrieve_with_scores(self, query: str) -> List[Dict]:
        """检索并返回结构化结果"""
        nodes = self.retrieve(query)
        results = []
        for node in nodes:
            results.append({
                "content": node.text,
                "score": round(node.score, 4) if node.score else 0,
                "file_name": node.metadata.get("file_name", "未知"),
            })
        return results


class QueryExpander:
    """查询扩展器：优化检索词，提升召回率"""
    
    def __init__(self, llm):
        self.llm = llm
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。
请为以下查询生成3个相关的扩展查询词，用于提升检索召回率。

要求：
1. 保持原查询的核心意图
2. 使用同义词、相关概念
3. 每个扩展查询用换行分隔

示例：
原查询：雇主责任险的保障范围
扩展查询：
雇主责任险包含哪些赔偿项目
雇主责任险理赔条件
雇主责任险承保范围

原查询：{query}
扩展查询："""),
        ])
        self.chain = self.expansion_prompt | llm | StrOutputParser()
    
    def expand(self, query: str) -> List[str]:
        """扩展查询词"""
        try:
            result = self.chain.invoke({"query": query})
            queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
            return [query] + queries[:3]
        except Exception as e:
            print(f"[查询扩展] 失败: {e}，使用原查询")
            return [query]


# ============================================================
# 第二部分：LangChain 编排层（Agent + 工具 + 记忆）
# ============================================================

class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_history: int = 5):
        self.history: List[Tuple[str, str]] = []
        self.max_history = max_history
    
    def add_turn(self, user_query: str, ai_response: str):
        """添加一轮对话"""
        self.history.append((user_query, ai_response))
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        """获取对话历史上下文"""
        if not self.history:
            return "无历史对话"
        
        context_parts = []
        for i, (query, response) in enumerate(self.history[-self.max_history:], 1):
            context_parts.append(f"对话{i}:\n用户: {query}\nAI: {response}")
        
        return "\n\n".join(context_parts)
    
    def clear(self):
        """清空对话历史"""
        self.history.clear()


def create_qa_chain(llm):
    """用 LangChain LCEL 构建问答链（支持对话历史）"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的保险产品顾问。
根据以下检索到的文档内容和对话历史回答用户的问题。
如果文档中没有相关信息，请如实说明。请用中文回复。

对话历史:
{history}

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


def create_evaluation_chain(llm):
    """用 LangChain LCEL 构建质量评估链"""
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个答案质量评估专家。
请从以下维度评估答案质量（0-100分）：

1. 准确性（40分）：答案是否准确反映了文档内容
2. 完整性（30分）：答案是否全面覆盖了问题要点
3. 相关性（30分）：答案是否与问题高度相关

请输出JSON格式：
{{"accuracy": 分数, "completeness": 分数, "relevance": 分数, "total": 总分, "comment": "简短评价"}}

文档内容:
{context}

问题: {question}

答案: {answer}

评估结果:"""),
        ("human", "请评估答案质量")
    ])
    return eval_prompt | llm | StrOutputParser()


@tool
def semantic_search(query: str, top_k: int = 5) -> str:
    """语义检索工具：从文档库中检索相关内容。

    参数:
        query: 检索查询词
        top_k: 返回结果数量（默认5）
    返回:
        检索到的文档内容
    """
    global hybrid_retriever
    results = hybrid_retriever.retrieve_with_scores(query, top_k)
    
    if not results:
        return "未检索到相关文档内容。"
    
    output_parts = []
    for i, res in enumerate(results, 1):
        preview = res["content"][:200]
        output_parts.append(
            f"[片段{i}] [来源: {res['file_name']}] [相似度: {res['score']}]\n{preview}"
        )
    
    return "\n\n".join(output_parts)


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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- 模拟发送邮件 ---")
    print(f"  收件人: {recipient}")
    print(f"  主题:   {subject}")
    print(f"  时间:   {timestamp}")
    print(f"  正文长度: {len(content)} 字")
    print(f"--- 邮件发送成功 ---\n")
    return f"邮件已于 {timestamp} 成功发送至 {recipient}，主题: {subject}"


# ============================================================
# 第三部分：主流程 —— 高级检索 + Agent + 记忆 + 评估
# ============================================================

class MultiFileQAAgent:
    """多文件智能问答Agent - LangChain + LlamaIndex 高级整合版"""
    
    def __init__(self):
        print("=" * 60)
        print("初始化多文件智能问答Agent v2.0")
        print("=" * 60)
        
        # 1. 初始化 LlamaIndex 数据层
        print("\n[1/5] 配置 LlamaIndex...")
        self.llamaindex_llm, self.embed_model = setup_llamaindex()
        
        print("[2/5] 构建/加载索引...")
        self.index = build_llamaindex_index()
        if self.index is None:
            raise ValueError("无法创建索引，程序退出")
        
        print("[3/5] 初始化混合检索器...")
        global hybrid_retriever
        hybrid_retriever = HybridRetriever(self.index, top_k=5)
        
        # 2. 初始化 LangChain 编排层
        print("[4/5] 初始化 LangChain...")
        self.langchain_llm = ChatTongyi(
            model_name="deepseek-v3.2",
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=0.1
        )
        
        # 3. 创建功能链
        self.qa_chain = create_qa_chain(self.langchain_llm)
        self.report_chain = create_report_chain(self.langchain_llm)
        self.eval_chain = create_evaluation_chain(self.langchain_llm)
        
        # 4. 创建辅助组件
        self.query_expander = QueryExpander(self.langchain_llm)
        self.memory = ConversationMemory(max_history=5)
        
        print("[5/5] 初始化完成\n")
    
    def chat(self, query: str) -> str:
        """对话接口（支持多轮对话）"""
        print(f"\n{'=' * 60}")
        print(f"用户: {query}")
        print(f"{'=' * 60}")
        
        # 1. 查询扩展
        print("\n[检索优化] 正在扩展查询词...")
        expanded_queries = self.query_expander.expand(query)
        print(f"扩展查询: {len(expanded_queries)} 个")
        for i, q in enumerate(expanded_queries, 1):
            print(f"  {i}. {q}")
        
        # 2. 混合检索
        print("\n[混合检索] 向量检索 + BM25 + RRF融合...")
        all_nodes = []
        for eq in expanded_queries:
            nodes = hybrid_retriever.retrieve(eq)
            all_nodes.extend(nodes)
        
        # 去重（按内容）
        unique_nodes = {}
        for node in all_nodes:
            key = node.text[:100]
            if key not in unique_nodes:
                unique_nodes[key] = node
        
        final_nodes = list(unique_nodes.values())[:5]
        
        print(f"召回 {len(final_nodes)} 个文档片段:")
        for i, node in enumerate(final_nodes, 1):
            score = round(node.score, 4) if node.score else 0
            preview = node.text[:100]
            print(f"  [{i}] [来源: {node.metadata.get('file_name', '未知')}] [相似度: {score}]")
            print(f"      {preview}...")
        
        # 3. 构建上下文
        context = "\n\n".join(
            f"[来源: {node.metadata.get('file_name', '未知')}]\n{node.text}"
            for node in final_nodes
        )
        
        # 4. 生成回答（含对话历史）
        print("\n[生成回答] 正在生成回答...")
        history = self.memory.get_context()
        answer = self.qa_chain.invoke({
            "context": context,
            "question": query,
            "history": history
        })
        
        print(f"\n{'=' * 60}")
        print(f"AI: {answer}")
        print(f"{'=' * 60}")
        
        # 5. 更新对话记忆
        self.memory.add_turn(query, answer)
        
        return answer
    
    def analyze(self, query: str) -> Dict:
        """深度分析接口（生成报告 + 质量评估）"""
        print(f"\n{'=' * 60}")
        print(f"深度分析: {query}")
        print(f"{'=' * 60}")
        
        # 1. 检索
        nodes = hybrid_retriever.retrieve(query)
        context = "\n\n".join(
            f"[来源: {node.metadata.get('file_name', '未知')}]\n{node.text}"
            for node in nodes
        )
        
        # 2. 生成回答
        answer = self.qa_chain.invoke({
            "context": context,
            "question": query,
            "history": "无历史对话"
        })
        
        # 3. 生成报告
        print("\n[报告生成] 正在生成分析报告...")
        sources = ", ".join(set(
            node.metadata.get('file_name', '未知') for node in nodes
        ))
        report = self.report_chain.invoke({
            "question": query,
            "answer": answer,
            "sources": sources
        })
        
        print(f"\n{'=' * 60}")
        print(f"分析报告:\n{report}")
        print(f"{'=' * 60}")
        
        # 4. 质量评估
        print("\n[质量评估] 正在评估答案质量...")
        eval_result = self.eval_chain.invoke({
            "context": context,
            "question": query,
            "answer": answer
        })
        
        print(f"\n评估结果: {eval_result}")
        
        # 5. 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"分析报告_{timestamp}.txt"
        full_report = f"问题: {query}\n\n回答:\n{answer}\n\n{report}\n\n评估结果:\n{eval_result}"
        save_report.invoke({"filename": filename, "content": full_report})
        
        return {
            "answer": answer,
            "report": report,
            "evaluation": eval_result,
            "sources": sources
        }


# ============================================================
# 主入口
# ============================================================

def main():
    """
    完整流程:
    1. LlamaIndex 建索引 + 混合检索
    2. LangChain 问答链（支持对话历史）
    3. 查询扩展 + 质量评估
    4. 工具调用保存报告 + 发送邮件
    """
    # 初始化系统
    agent = MultiFileQAAgent()
    
    print("=" * 60)
    print("欢迎使用多文件智能问答Agent v2.0")
    print("=" * 60)
    print("\n功能说明:")
    print("  - 直接输入问题: 进行智能问答")
    print("  - 输入 'analyze <问题>': 深度分析模式")
    print("  - 输入 'history': 查看对话历史")
    print("  - 输入 'clear': 清空对话历史")
    print("  - 输入 'quit': 退出程序")
    print("=" * 60)
    
    # 交互式对话
    while True:
        user_input = input("\n请输入: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\n感谢使用，再见！")
            break
        
        if user_input.lower() == 'history':
            print(f"\n对话历史:\n{agent.memory.get_context()}")
            continue
        
        if user_input.lower() == 'clear':
            agent.memory.clear()
            print("\n对话历史已清空")
            continue
        
        if user_input.lower().startswith('analyze '):
            query = user_input[8:].strip()
            agent.analyze(query)
        else:
            agent.chat(user_input)


if __name__ == "__main__":
    main()