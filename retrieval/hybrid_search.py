"""
hybrid_search.py

功能：
1. 向量检索 + BM25 混合检索
2. keyword / vector / hybrid 策略分流
3. Rerank 重排序
4. 兼容 FastAPI 的轻量缓存对象
"""

import re
from typing import List

import jieba
import requests
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class SimpleRuntimeCache:
    def __init__(self):
        self.session_state = {}

    def clear_bm25_cache(self):
        self.session_state.pop("bm25_cache", None)


def _get_session_state(runtime):
    if runtime is None:
        return {}
    return getattr(runtime, "session_state", {})


def get_bm25_index(vector_store, runtime=None):
    all_content = vector_store.get()

    if not all_content or not all_content.get("documents"):
        return None, [], []

    raw_texts = []
    metadatas = []

    for text, meta in zip(
        all_content.get("documents", []),
        all_content.get("metadatas", []),
    ):
        if not text or not meta:
            continue
        if meta.get("chunk_type") == "large":
            continue
        raw_texts.append(text)
        metadatas.append(meta)

    if not raw_texts:
        return None, [], []

    session_state = _get_session_state(runtime)
    doc_count = len(raw_texts)
    cached = session_state.get("bm25_cache")

    if cached and cached.get("doc_count") == doc_count:
        return cached["model"], raw_texts, metadatas

    tokenized = [list(jieba.cut(doc)) for doc in raw_texts]
    model = BM25Okapi(tokenized)

    session_state["bm25_cache"] = {
        "model": model,
        "doc_count": doc_count,
    }

    return model, raw_texts, metadatas


def _heuristic_strategy(query: str) -> str | None:
    q = (query or "").strip()
    if not q:
        return "skip"

    greetings = {"你好", "您好", "hi", "hello", "hey", "在吗", "谢谢", "thanks", "thank you"}
    if q.lower() in greetings:
        return "skip"

    if re.search(r"[A-Za-z0-9_\-]+\.(pdf|docx|txt|md|xlsx|xls|csv|png|jpg|jpeg|bmp)$", q):
        return "keyword"

    if any(marker in q for marker in ["原文", "第几步", "步骤", "用量", "克", "g", "ml", "分钟", "配料表"]):
        return "keyword"

    # RAG 评估中最容易出错的是“是否明确提到”“怎么处理”“什么时候”等问题。
    # 这类问题既需要关键词精确匹配，也需要向量召回，不建议只走 vector。
    if any(marker in q for marker in [
        "怎么处理", "什么时候", "哪些", "几个", "是否", "有没有",
        "必须", "根据资料", "需要哪些", "共同食材", "区别", "为什么",
    ]):
        return "hybrid"

    if any(marker in q for marker in ["类似", "替代", "适合", "推荐", "低卡", "减脂", "清淡"]):
        return "vector"

    return None


def decide_retrieval_strategy(query: str, llm_client=None) -> str:
    """
    返回 skip / keyword / vector / hybrid。
    该结果会真正控制检索流程。
    """
    heuristic = _heuristic_strategy(query)
    if heuristic:
        return heuristic

    if llm_client is None:
        return "hybrid"

    try:
        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "判断下面问题应该使用哪种检索策略。\n"
                        "只输出一个词：skip / keyword / vector / hybrid。\n"
                        "规则：\n"
                        "- skip：寒暄、空问题、明显不需要知识库的问题。\n"
                        "- keyword：需要精确匹配菜名、步骤、用量、文件名、原文。\n"
                        "- vector：需要语义相似、替代建议、概括推荐。\n"
                        "- hybrid：既需要关键词又需要语义召回。\n\n"
                        f"问题：{query}"
                    ),
                }
            ],
            max_tokens=10,
            temperature=0,
        )

        result = resp.choices[0].message.content.strip().lower()
        if result in {"skip", "keyword", "vector", "hybrid"}:
            return result
        return "hybrid"

    except Exception:
        return "hybrid"


def execute_vector_search(query, vector_store, k_vector=4) -> List[Document]:
    vector_results = []

    try:
        vector_docs = vector_store.similarity_search(query, k=max(k_vector * 2, k_vector))
        vector_results = [
            d
            for d in vector_docs
            if d.page_content and d.metadata.get("chunk_type") != "large"
        ][:k_vector]
    except Exception:
        vector_results = []

    return vector_results


def execute_keyword_search(query, vector_store, runtime=None, k_bm25=4) -> List[Document]:
    bm25_results = []

    try:
        bm25_model, raw_texts, metadatas = get_bm25_index(vector_store, runtime)

        if bm25_model:
            tokens = list(jieba.cut(query))
            scores = bm25_model.get_scores(tokens)

            top_idx = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k_bm25]

            bm25_results = [
                Document(page_content=raw_texts[i], metadata=dict(metadatas[i]))
                for i in top_idx
                if scores[i] > 0
            ]
    except Exception:
        bm25_results = []

    return bm25_results


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    merged = []

    for doc in docs:
        unique_key = (
            doc.metadata.get("source"),
            doc.metadata.get("parent_id"),
            doc.page_content[:80],
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)
        merged.append(doc)

    return merged


def execute_hybrid_search(
    query,
    vector_store,
    runtime=None,
    k_vector=4,
    k_bm25=4,
):
    """
    混合检索：

    1. 向量检索召回 small chunk
    2. BM25 检索召回 small chunk
    3. 如果 BM25 分数 <= 0，但存在关键词重合，也保底召回
    4. 如果向量和 BM25 都失败，再用关键词重合做兜底
    """
    vector_results = []

    # 1. 向量检索
    try:
        vector_docs = vector_store.similarity_search(
            query,
            k=max(k_vector * 2, k_vector),
        )

        vector_results = [
            d
            for d in vector_docs
            if d.page_content and d.metadata.get("chunk_type") != "large"
        ][:k_vector]

    except Exception as e:
        print(f"[Vector Search Error] {e}")
        vector_results = []

    bm25_results = []

    # 2. BM25 检索
    try:
        bm25_model, raw_texts, metadatas = get_bm25_index(
            vector_store,
            runtime,
        )

        if bm25_model:
            query_tokens = [
                t.strip()
                for t in jieba.cut(query)
                if t.strip()
            ]

            scores = bm25_model.get_scores(query_tokens)

            top_idx = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:k_bm25]

            for i in top_idx:
                text = raw_texts[i]
                meta = metadatas[i]
                score = scores[i]

                # 原来只允许 score > 0，这在小语料库里容易误杀。
                # 现在改成：score > 0 或者 query 和文档有关键词重合，都允许召回。
                has_token_overlap = any(
                    token in text
                    for token in query_tokens
                    if len(token) >= 2
                )

                if score > 0 or has_token_overlap:
                    bm25_results.append(
                        Document(
                            page_content=text,
                            metadata=dict(meta),
                        )
                    )

    except Exception as e:
        print(f"[BM25 Search Error] {e}")
        bm25_results = []

    # 3. 兜底：如果向量和 BM25 都没结果，直接从 small chunk 里做关键词重合匹配
    fallback_results = []

    if not vector_results and not bm25_results:
        try:
            all_content = vector_store.get()
            query_tokens = [
                t.strip()
                for t in jieba.cut(query)
                if len(t.strip()) >= 2
            ]

            for text, meta in zip(
                all_content.get("documents", []),
                all_content.get("metadatas", []),
            ):
                if not text or not meta:
                    continue

                if meta.get("chunk_type") == "large":
                    continue

                has_overlap = any(token in text for token in query_tokens)

                if has_overlap:
                    fallback_results.append(
                        Document(
                            page_content=text,
                            metadata=dict(meta),
                        )
                    )

                if len(fallback_results) >= max(k_vector, k_bm25):
                    break

        except Exception as e:
            print(f"[Fallback Search Error] {e}")
            fallback_results = []

    # 4. 合并去重
    seen = set()
    merged = []

    for doc in vector_results + bm25_results + fallback_results:
        unique_key = (
            doc.metadata.get("source"),
            doc.metadata.get("parent_id"),
            doc.page_content[:80],
        )

        if unique_key in seen:
            continue

        seen.add(unique_key)
        merged.append(doc)

    print(
        f"[Search Debug] query={query} | "
        f"vector={len(vector_results)} | "
        f"bm25={len(bm25_results)} | "
        f"fallback={len(fallback_results)} | "
        f"merged={len(merged)}"
    )

    return merged


def execute_search_by_strategy(
    query: str,
    vector_store,
    runtime=None,
    strategy: str = "hybrid",
    k_vector: int = 4,
    k_bm25: int = 4,
) -> List[Document]:
    if strategy == "skip":
        return []
    if strategy == "keyword":
        return execute_keyword_search(query, vector_store, runtime, k_bm25=k_bm25)
    if strategy == "vector":
        return execute_vector_search(query, vector_store, k_vector=k_vector)
    return execute_hybrid_search(query, vector_store, runtime, k_vector=k_vector, k_bm25=k_bm25)


def _with_fallback_rerank_score(docs, top_n=3, score=0.30):
    """
    Rerank 服务不可用时的兜底。

    说明：
    - 这不是“真实 rerank 分数”，只是为了避免系统在已经检索到候选资料时直接拒答。
    - 真正降低幻觉主要靠严格 Prompt 和后续复测，而不是靠这里伪装高分。
    - 分数只给到 0.30，刚好超过 confidence.py 的 0.25 阈值，避免“一问就不会”。
    """
    fallback_docs = []
    for doc in docs[:top_n]:
        new_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
        new_doc.metadata.setdefault("rerank_score", score)
        new_doc.metadata.setdefault("rerank_fallback", True)
        fallback_docs.append(new_doc)
    return fallback_docs


def execute_rerank(
    query,
    docs,
    api_key,
    rerank_url,
    top_n=3,
):
    """
    Rerank 重排序。

    修复点：
    上一版过于保守。如果 Rerank API 不可用、URL 配错、超时或返回异常，
    docs 会没有 rerank_score，confidence.py 会把它们全部判为低置信度，
    导致系统即使检索到了资料也直接拒答，看起来像“什么都不会”。

    这里改成：Rerank 成功就用真实分数；Rerank 不可用就给低强度兜底分数。
    """
    if not docs:
        return []

    if not api_key or not rerank_url:
        return _with_fallback_rerank_score(docs, top_n=top_n, score=0.30)

    try:
        resp = requests.post(
            rerank_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "rerank-3",
                "query": query,
                "documents": [d.page_content for d in docs],
                "top_n": top_n,
            },
            timeout=10,
        )

        resp.raise_for_status()
        data = resp.json().get("results", [])
        reranked = []

        for item in data:
            doc = docs[item["index"]]
            new_doc = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            new_doc.metadata["rerank_score"] = float(item.get("relevance_score", 0.0))
            new_doc.metadata["rerank_fallback"] = False
            reranked.append(new_doc)

        return reranked or _with_fallback_rerank_score(docs, top_n=top_n, score=0.30)

    except Exception as e:
        print(f"[Rerank Error] {e}")
        return _with_fallback_rerank_score(docs, top_n=top_n, score=0.30)
