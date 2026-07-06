"""
evaluator.py

功能：
1. 检索质量评估
2. 答案相关性评估
3. 简单RAG指标计算
"""

from typing import Dict, List


# 检索评分
def evaluate_retrieval(retrieved_docs: List) -> Dict:
    if not retrieved_docs:
        return {"retrieval_score": 0.0, "doc_count": 0}

    scores = [doc.metadata.get("rerank_score", 0.0) for doc in retrieved_docs]

    return {
        "retrieval_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "doc_count": len(retrieved_docs),
    }


# 忠实度评估
def evaluate_faithfulness(answer: str, contexts: List) -> float:
    if not contexts or not answer:
        return 0.0

    context_text = "\n".join(d.page_content[:500] for d in contexts)

    answer_tokens = set(answer.split())
    context_tokens = set(context_text.split())

    if not answer_tokens:
        return 0.0

    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 4)


# 上下文精度
def evaluate_context_precision(retrieved_docs: List) -> float:
    if not retrieved_docs:
        return 0.0

    scores = [d.metadata.get("rerank_score", 0) for d in retrieved_docs]

    return round(sum(scores) / len(scores), 4) if scores else 0.0


# 答案相关性
def evaluate_answer_relevance(question: str, answer: str) -> float:
    q = set(question.split())
    a = set(answer.split())

    if not q:
        return 0.0

    return round(len(q & a) / len(q), 4)


# 综合评估
def evaluate_all(question: str, answer: str, contexts: List) -> Dict:
    retrieval = evaluate_retrieval(contexts)

    return {
        "retrieval_score": retrieval["retrieval_score"],
        "retrieved_docs": retrieval["doc_count"],
        "faithfulness": evaluate_faithfulness(answer, contexts),
        "context_precision": evaluate_context_precision(contexts),
        "answer_relevance": evaluate_answer_relevance(question, answer),
    }