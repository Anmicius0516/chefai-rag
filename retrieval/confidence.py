"""
retrieval/confidence.py

负责：
1. Retrieval Confidence
2. Refusal Decision
3. Iterative Retrieval

本版修改：
- 阈值稍微降低，更适合本地小知识库 / TXT 测试。
- 如果没有 rerank_score，仍然不放行，避免无依据回答。
"""

from typing import List, Tuple


FAITHFULNESS_THRESHOLD = 0.20
SUPPLEMENT_THRESHOLD = 0.10


def check_retrieval_confidence(
    docs: List,
    threshold: float = FAITHFULNESS_THRESHOLD,
) -> Tuple[bool, float]:
    """
    根据 rerank_score 判断检索结果是否可信。

    返回：
        (是否可信, 最大 rerank_score)

    重要：
        如果 docs 中没有任何 rerank_score，说明没有完成可靠重排，
        这里必须低置信度处理，不能返回 True, 1.0。
    """
    if not docs:
        return False, 0.0

    scores = []
    for doc in docs:
        raw_score = getattr(doc, "metadata", {}).get("rerank_score")
        if raw_score is None:
            continue
        try:
            scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue

    if not scores:
        return False, 0.0

    max_score = max(scores)
    return max_score >= threshold, max_score


def need_supplement_retrieval(
    score: float,
    threshold: float = FAITHFULNESS_THRESHOLD,
    supplement_threshold: float = SUPPLEMENT_THRESHOLD,
) -> bool:
    return supplement_threshold <= score < threshold


def execute_iterative_retrieval(
    query: str,
    initial_docs: list,
    llm_client,
    hybrid_search_fn,
    rerank_fn,
):
    """
    如果初次检索分数在补充检索区间，就让 LLM 生成 2 个子问题再检索一次。
    """
    is_confident, score = check_retrieval_confidence(initial_docs)

    if is_confident or score < SUPPLEMENT_THRESHOLD:
        return initial_docs

    try:
        prompt = (
            f"原始问题：\n{query}\n\n"
            "请从不同角度生成2个更精确的食谱或文档检索子问题。\n"
            "要求：\n"
            "1. 保留原问题中的关键名词\n"
            "2. 可以补充同义词或相关表达\n"
            "3. 每行一个\n"
            "4. 不要编号"
        )

        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )

        sub_queries = [
            q.strip()
            for q in resp.choices[0].message.content.split("\n")
            if q.strip()
        ][:2]

    except Exception:
        return initial_docs

    seen = {doc.page_content for doc in initial_docs}
    all_docs = list(initial_docs)

    for sub_query in sub_queries:
        extra_docs = hybrid_search_fn(sub_query, k_vector=3, k_bm25=3)
        for doc in extra_docs:
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)

    return rerank_fn(query, all_docs, top_n=5)
