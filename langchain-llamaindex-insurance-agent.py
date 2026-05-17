#!/usr/bin/env python
# coding: utf-8
"""
商业保险智能问答Agent v3.0 - LangChain + LlamaIndex 深度整合版

保险业务场景：
  - 雇主责任险、企业团体意外险、商业综合责任保险
  - 财产一切险、施工保、装修保、交通意外险
  - 医疗救援、疾病身故保险

核心功能：
  - 混合检索：向量检索 + BM25关键词检索 + RRF融合
  - 多Agent协作：咨询Agent + 对比Agent + 理赔Agent + 风险评估Agent
  - 智能路由：根据问题类型自动选择处理策略
  - 产品对比：多维度保险产品对比分析
  - 理赔推荐：根据事故场景推荐最优理赔方案
  - 风险评估：企业风险等级评估与保险配置建议
  - 条款解读：专业术语解释、免责条款提取
  - 对话记忆：支持多轮对话上下文
  - 质量评估：答案质量自动评分

架构思路：
  - LlamaIndex 负责「数据层」：文档处理、混合检索、查询转换
  - LangChain  负责「编排层」：多Agent协作、工具调用、链式编排、记忆管理
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
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document as LCDocument
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")


# ============================================================
# 第一部分：LlamaIndex 数据层（保险文档专属优化）
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
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    return llm, embed_model


def build_llamaindex_index(file_dir: str = './docs', persist_dir: str = './combined_storage'):
    """用 LlamaIndex 加载保险文档并构建向量索引"""
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

    print(f"[LlamaIndex] 加载了 {len(documents)} 个保险文档，正在构建索引...")
    
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"[LlamaIndex] 文档分块完成，共 {len(nodes)} 个片段")
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"[LlamaIndex] 索引已保存到 {persist_dir}")
    return index


class InsuranceHybridRetriever:
    """保险文档混合检索器"""
    
    def __init__(self, index, top_k: int = 5):
        self.index = index
        self.top_k = top_k
        
        self.vector_retriever = index.as_retriever(
            similarity_top_k=top_k * 2
        )
        
        self.bm25_retriever = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=top_k * 2
        )
        
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            similarity_top_k=top_k,
            mode="reciprocal_rerank",
            use_async=False,
            num_queries=3,
            verbose=False
        )
        
        self.postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.4),
        ]
    
    def retrieve(self, query: str) -> List[NodeWithScore]:
        """执行混合检索"""
        nodes = self.fusion_retriever.retrieve(query)
        for processor in self.postprocessors:
            nodes = processor.postprocess_nodes(nodes)
        return nodes[:self.top_k]
    
    def retrieve_by_product(self, product_name: str, query: str) -> List[NodeWithScore]:
        """按产品名称过滤检索"""
        all_nodes = self.retrieve(query)
        filtered = [
            node for node in all_nodes 
            if product_name.lower() in node.metadata.get("file_name", "").lower()
        ]
        return filtered if filtered else all_nodes[:3]
    
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


class InsuranceQueryExpander:
    """保险查询扩展器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是保险行业查询优化专家。
请为以下保险相关查询生成3个相关的扩展查询词。

扩展策略：
1. 同义词替换（如：保障范围→承保范围→保险责任）
2. 相关概念（如：理赔→理赔流程→理赔材料→理赔时效）
3. 具体化/抽象化

示例：
原查询：雇主责任险的保障范围
扩展查询：
雇主责任险包含哪些赔偿项目
雇主责任险保险责任条款
雇主责任险承保范围和免责条款

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
            print(f"[查询扩展] 失败: {e}")
            return [query]


# ============================================================
# 第二部分：LangChain 编排层（多Agent协作 + 保险专属工具）
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


class InsuranceRouter:
    """保险问题智能路由器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是保险业务问题分类专家。
请将用户问题分类到以下保险业务场景之一：

CONSULTATION: 保险产品咨询（保障范围、保费、条款解读等）
COMPARISON: 保险产品对比（多个产品对比、优劣势分析）
CLAIM: 理赔相关（理赔流程、理赔材料、理赔条件）
RISK_ASSESSMENT: 风险评估（企业风险评估、保险配置建议）
POLICY_INTERPRETATION: 条款解读（专业术语解释、免责条款）

仅输出分类结果，不要解释。

用户问题: {query}
分类结果:"""),
        ])
        self.chain = self.router_prompt | llm | StrOutputParser()
    
    def route(self, query: str) -> str:
        """路由分类"""
        try:
            result = self.chain.invoke({"query": query}).strip().upper()
            category_map = {
                "CONSULTATION": "consultation",
                "COMPARISON": "comparison",
                "CLAIM": "claim",
                "RISK_ASSESSMENT": "risk_assessment",
                "POLICY_INTERPRETATION": "policy_interpretation",
            }
            return category_map.get(result, "consultation")
        except Exception as e:
            print(f"[智能路由] 失败: {e}")
            return "consultation"


def create_consultation_chain(llm):
    """保险咨询问答链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是资深保险顾问，精通各类商业保险产品。
根据检索到的保险文档内容，专业、准确地回答用户问题。

回答要求：
1. 引用具体条款内容
2. 标注信息来源文档
3. 语言专业但通俗易懂
4. 如文档无相关信息，如实说明

对话历史:
{history}

检索到的保险文档:
{context}"""),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()


def create_comparison_chain(llm):
    """保险产品对比链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是保险产品对比分析专家。
根据检索到的多个保险产品文档，进行多维度对比分析。

对比维度：
1. 保障范围对比
2. 保险金额与保费
3. 免责条款对比
4. 理赔条件与流程
5. 适用场景与人群
6. 产品优劣势

检索到的文档内容:
{context}"""),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()


def create_claim_chain(llm):
    """理赔推荐链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是保险理赔专家。
根据用户描述的事故场景和检索到的保险条款，提供理赔方案。

输出格式：
1. 适用保险产品推荐
2. 理赔条件判断
3. 理赔流程指引
4. 所需材料清单
5. 注意事项与时效要求

检索到的保险条款:
{context}"""),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()


def create_risk_assessment_chain(llm):
    """风险评估链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是企业风险评估与保险配置专家。
根据用户提供的企业信息和检索到的保险产品，生成风险评估报告。

报告结构：
1. 企业风险等级评估（高/中/低）
2. 主要风险点识别
3. 推荐保险组合方案
4. 各保险产品保障说明
5. 保费预算建议
6. 风险防范措施

检索到的保险产品:
{context}"""),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()


def create_policy_interpretation_chain(llm):
    """条款解读链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是保险条款解读专家。
请对检索到的保险条款进行专业解读。

解读要求：
1. 专业术语解释
2. 核心条款提取
3. 免责条款重点说明
4. 实际案例说明
5. 注意事项

检索到的条款内容:
{context}"""),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()


def create_report_chain(llm):
    """报告生成链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是保险分析报告生成专家。
根据问答记录生成结构化分析报告。

报告结构：
1. 概述
2. 核心要点
3. 适用场景
4. 注意事项
5. 建议与总结

问答记录:
问题: {question}
回答: {answer}
来源: {sources}"""),
        ("human", "请生成分析报告")
    ])
    return prompt | llm | StrOutputParser()


def create_evaluation_chain(llm):
    """质量评估链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是保险答案质量评估专家。
从以下维度评估答案质量（0-100分）：

1. 准确性（40分）：是否准确反映保险条款
2. 完整性（30分）：是否全面覆盖问题要点
3. 专业性（30分）：是否使用专业术语和表述

输出JSON格式：
{{"accuracy": 分数, "completeness": 分数, "professionalism": 分数, "total": 总分, "comment": "评价"}}

保险文档:
{context}

问题: {question}

答案: {answer}

评估结果:"""),
        ("human", "请评估")
    ])
    return prompt | llm | StrOutputParser()


# ============================================================
# 第三部分：保险专属工具定义
# ============================================================

@tool
def search_insurance(query: str, top_k: int = 5) -> str:
    """保险文档检索工具：检索相关保险条款和产品说明。

    参数:
        query: 检索查询词
        top_k: 返回结果数量
    返回:
        检索到的保险文档内容
    """
    global hybrid_retriever
    results = hybrid_retriever.retrieve_with_scores(query)
    
    if not results:
        return "未检索到相关保险文档。"
    
    output_parts = []
    for i, res in enumerate(results, 1):
        preview = res["content"][:250]
        output_parts.append(
            f"[片段{i}] [产品: {res['file_name']}] [相关度: {res['score']}]\n{preview}"
        )
    
    return "\n\n".join(output_parts)


@tool
def compare_insurance_products(product_names: str, aspect: str = "保障范围") -> str:
    """保险产品对比工具：对比多个保险产品的特定维度。

    参数:
        product_names: 产品名称列表（用逗号分隔）
        aspect: 对比维度（保障范围/保费/理赔条件/免责条款）
    返回:
        对比分析结果
    """
    global hybrid_retriever
    products = [p.strip() for p in product_names.split(',')]
    
    comparisons = []
    for product in products:
        nodes = hybrid_retriever.retrieve_by_product(product, aspect)
        content = "\n".join([n.text[:200] for n in nodes])
        comparisons.append(f"【{product}】\n{content}")
    
    return "\n\n".join(comparisons)


@tool
def get_claim_guide(scenario: str) -> str:
    """理赔指引工具：根据事故场景提供理赔指引。

    参数:
        scenario: 事故场景描述
    返回:
        理赔指引内容
    """
    global hybrid_retriever
    query = f"{scenario} 理赔流程 理赔材料"
    nodes = hybrid_retriever.retrieve(query)
    
    if not nodes:
        return "未检索到相关理赔指引。"
    
    return "\n\n".join([
        f"[来源: {n.metadata.get('file_name', '未知')}]\n{n.text[:300]}"
        for n in nodes[:3]
    ])


@tool
def explain_term(term: str) -> str:
    """保险术语解释工具：解释保险专业术语。

    参数:
        term: 保险术语
    返回:
        术语解释
    """
    global hybrid_retriever
    query = f"{term} 定义 含义 解释"
    nodes = hybrid_retriever.retrieve(query)
    
    if not nodes:
        return f"未检索到'{term}'的相关解释。"
    
    return "\n\n".join([
        f"[来源: {n.metadata.get('file_name', '未知')}]\n{n.text[:200]}"
        for n in nodes[:2]
    ])


@tool
def save_report(filename: str, content: str) -> str:
    """保存报告到本地文件。

    参数:
        filename: 文件名
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
    """发送邮件工具。

    参数:
        recipient: 收件人邮箱
        subject: 邮件主题
        content: 邮件正文
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
    return f"邮件已于 {timestamp} 成功发送至 {recipient}"


# ============================================================
# 第四部分：主系统 - 多Agent协作 + 智能路由
# ============================================================

class InsuranceQAAgent:
    """商业保险智能问答Agent - v3.0"""
    
    def __init__(self):
        print("=" * 60)
        print("商业保险智能问答Agent v3.0")
        print("LangChain + LlamaIndex 深度整合版")
        print("=" * 60)
        
        # 1. 初始化 LlamaIndex 数据层
        print("\n[1/6] 配置 LlamaIndex...")
        self.llamaindex_llm, self.embed_model = setup_llamaindex()
        
        print("[2/6] 构建/加载保险文档索引...")
        self.index = build_llamaindex_index()
        if self.index is None:
            raise ValueError("无法创建索引")
        
        print("[3/6] 初始化混合检索器...")
        global hybrid_retriever
        hybrid_retriever = InsuranceHybridRetriever(self.index, top_k=5)
        
        # 2. 初始化 LangChain 编排层
        print("[4/6] 初始化 LangChain...")
        self.langchain_llm = ChatTongyi(
            model_name="deepseek-v3.2",
            dashscope_api_key=DASHSCOPE_API_KEY,
            temperature=0.1
        )
        
        # 3. 创建功能链
        print("[5/6] 创建功能链...")
        self.chains = {
            "consultation": create_consultation_chain(self.langchain_llm),
            "comparison": create_comparison_chain(self.langchain_llm),
            "claim": create_claim_chain(self.langchain_llm),
            "risk_assessment": create_risk_assessment_chain(self.langchain_llm),
            "policy_interpretation": create_policy_interpretation_chain(self.langchain_llm),
            "report": create_report_chain(self.langchain_llm),
            "evaluation": create_evaluation_chain(self.langchain_llm),
        }
        
        # 4. 创建辅助组件
        self.router = InsuranceRouter(self.langchain_llm)
        self.query_expander = InsuranceQueryExpander(self.langchain_llm)
        self.memory = ConversationMemory(max_history=5)
        
        print("[6/6] 初始化完成\n")
    
    def chat(self, query: str) -> str:
        """对话接口（智能路由）"""
        print(f"\n{'=' * 60}")
        print(f"用户: {query}")
        print(f"{'=' * 60}")
        
        # 1. 智能路由分类
        category = self.router.route(query)
        print(f"\n[智能路由] 问题类型: {category}")
        
        # 2. 查询扩展
        print("[检索优化] 正在扩展查询词...")
        expanded_queries = self.query_expander.expand(query)
        print(f"扩展查询: {len(expanded_queries)} 个")
        
        # 3. 混合检索
        print("\n[混合检索] 向量+BM25+RRF融合...")
        all_nodes = []
        for eq in expanded_queries:
            nodes = hybrid_retriever.retrieve(eq)
            all_nodes.extend(nodes)
        
        # 去重
        unique_nodes = {}
        for node in all_nodes:
            key = node.text[:100]
            if key not in unique_nodes:
                unique_nodes[key] = node
        
        final_nodes = list(unique_nodes.values())[:5]
        
        print(f"召回 {len(final_nodes)} 个文档片段:")
        for i, node in enumerate(final_nodes, 1):
            score = round(node.score, 4) if node.score else 0
            preview = node.text[:80]
            print(f"  [{i}] [产品: {node.metadata.get('file_name', '未知')}] [相关度: {score}]")
            print(f"      {preview}...")
        
        # 4. 构建上下文
        context = "\n\n".join(
            f"[产品: {node.metadata.get('file_name', '未知')}]\n{node.text}"
            for node in final_nodes
        )
        
        # 5. 选择对应链并生成回答
        print(f"\n[生成回答] 使用 {category} 链...")
        chain = self.chains.get(category, self.chains["consultation"])
        
        history = self.memory.get_context()
        answer = chain.invoke({
            "context": context,
            "question": query,
            "history": history
        })
        
        print(f"\n{'=' * 60}")
        print(f"AI: {answer}")
        print(f"{'=' * 60}")
        
        # 6. 更新记忆
        self.memory.add_turn(query, answer)
        
        return answer
    
    def analyze(self, query: str) -> Dict:
        """深度分析接口"""
        print(f"\n{'=' * 60}")
        print(f"深度分析: {query}")
        print(f"{'=' * 60}")
        
        # 1. 检索
        nodes = hybrid_retriever.retrieve(query)
        context = "\n\n".join(
            f"[产品: {node.metadata.get('file_name', '未知')}]\n{node.text}"
            for node in nodes
        )
        
        # 2. 生成回答
        category = self.router.route(query)
        chain = self.chains.get(category, self.chains["consultation"])
        answer = chain.invoke({
            "context": context,
            "question": query,
            "history": "无历史对话"
        })
        
        # 3. 生成报告
        print("\n[报告生成] 正在生成分析报告...")
        sources = ", ".join(set(
            node.metadata.get('file_name', '未知') for node in nodes
        ))
        report = self.chains["report"].invoke({
            "question": query,
            "answer": answer,
            "sources": sources
        })
        
        print(f"\n{'=' * 60}")
        print(f"分析报告:\n{report}")
        print(f"{'=' * 60}")
        
        # 4. 质量评估
        print("\n[质量评估] 正在评估答案质量...")
        eval_result = self.chains["evaluation"].invoke({
            "context": context,
            "question": query,
            "answer": answer
        })
        
        print(f"\n评估结果: {eval_result}")
        
        # 5. 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"保险分析报告_{timestamp}.txt"
        full_report = f"问题: {query}\n\n回答:\n{answer}\n\n{report}\n\n评估结果:\n{eval_result}"
        save_report.invoke({"filename": filename, "content": full_report})
        
        return {
            "answer": answer,
            "report": report,
            "evaluation": eval_result,
            "sources": sources
        }
    
    def compare_products(self, product_names: List[str]) -> str:
        """保险产品对比"""
        print(f"\n{'=' * 60}")
        print(f"产品对比: {', '.join(product_names)}")
        print(f"{'=' * 60}")
        
        query = f"对比 {' '.join(product_names)} 的保障范围 保费 理赔条件 免责条款"
        nodes = hybrid_retriever.retrieve(query)
        context = "\n\n".join([n.text for n in nodes])
        
        comparison = self.chains["comparison"].invoke({
            "context": context,
            "question": f"请详细对比以下保险产品: {', '.join(product_names)}"
        })
        
        print(f"\n{'=' * 60}")
        print(f"对比结果:\n{comparison}")
        print(f"{'=' * 60}")
        
        return comparison
    
    def risk_assessment(self, company_info: str) -> str:
        """企业风险评估"""
        print(f"\n{'=' * 60}")
        print(f"风险评估: {company_info}")
        print(f"{'=' * 60}")
        
        nodes = hybrid_retriever.retrieve("企业保险 风险评估 保险配置")
        context = "\n\n".join([n.text for n in nodes])
        
        assessment = self.chains["risk_assessment"].invoke({
            "context": context,
            "question": f"请为以下企业进行风险评估和保险配置建议: {company_info}"
        })
        
        print(f"\n{'=' * 60}")
        print(f"风险评估报告:\n{assessment}")
        print(f"{'=' * 60}")
        
        return assessment


# ============================================================
# 主入口
# ============================================================

def main():
    """
    完整流程:
    1. LlamaIndex 建索引 + 混合检索
    2. LangChain 多Agent协作 + 智能路由
    3. 保险专属工具（产品对比/理赔指引/术语解释）
    4. 查询扩展 + 质量评估
    5. 工具调用保存报告 + 发送邮件
    """
    agent = InsuranceQAAgent()
    
    print("=" * 60)
    print("欢迎使用商业保险智能问答Agent v3.0")
    print("=" * 60)
    print("\n功能说明:")
    print("  - 直接输入问题: 智能问答（自动路由）")
    print("  - 输入 'analyze <问题>': 深度分析模式")
    print("  - 输入 'compare <产品1>,<产品2>': 产品对比")
    print("  - 输入 'risk <企业信息>': 风险评估")
    print("  - 输入 'history': 查看对话历史")
    print("  - 输入 'clear': 清空对话历史")
    print("  - 输入 'quit': 退出程序")
    print("=" * 60)
    
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
        
        if user_input.lower().startswith('compare '):
            products = user_input[8:].split(',')
            products = [p.strip() for p in products]
            agent.compare_products(products)
        
        elif user_input.lower().startswith('risk '):
            company_info = user_input[5:].strip()
            agent.risk_assessment(company_info)
        
        elif user_input.lower().startswith('analyze '):
            query = user_input[8:].strip()
            agent.analyze(query)
        
        else:
            agent.chat(user_input)


if __name__ == "__main__":
    main()