"""
chroma_manager.py

负责：
1. Embedding Service
2. Chroma 初始化
3. 文档入库
4. 数据源管理

说明：
- 不强依赖 Streamlit，方便同时被 FastAPI 后端和旧版 Streamlit Demo 复用。
- 入库时同时保存 large parent chunk 与 small child chunk。
- small chunk 用于检索定位，large chunk 用于回答上下文回溯。
"""

from typing import Callable, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from retrieval.parent_retriever import split_documents_hierarchical


# ─────────────────────────────────────────────
# Embedding Service
# ─────────────────────────────────────────────

class ZhipuEmbeddingService:
    """
    适配 LangChain Chroma 的智谱 Embedding 封装。
    """

    def __init__(self, client, model_name: str = "embedding-3"):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        return [
            item.embedding
            for item in response.data
        ]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
        )

        return response.data[0].embedding


# ─────────────────────────────────────────────
# Chroma 初始化
# ─────────────────────────────────────────────

def init_vector_store(db_path: str, embedding_service):
    """
    初始化 Chroma 向量库。

    这里不使用 @st.cache_resource，
    这样 FastAPI 后端也可以正常复用。
    """
    return Chroma(
        persist_directory=db_path,
        embedding_function=embedding_service,
    )


# ─────────────────────────────────────────────
# 文档入库
# ─────────────────────────────────────────────

_SYSTEM_SOURCES = {
    "system",
    "",
    "Database initialized.",
    "Database initialized successfully.",
}


def _try_clear_streamlit_bm25_cache():
    """
    兼容旧版 app.py。

    如果当前运行在 Streamlit 里，就清除 st.session_state 里的 bm25_cache。
    如果当前运行在 FastAPI 里，这个函数会静默跳过。
    """
    try:
        import streamlit as st

        if hasattr(st, "session_state"):
            st.session_state.pop("bm25_cache", None)

    except Exception:
        pass


def ingest_documents(
    vector_store,
    raw_docs: list[Document],
    source_name: str,
    clear_cache: Optional[Callable[[], None]] = None,
    return_detail: bool = False,
):
    """
    文档入库。

    默认返回 int，用来兼容旧版 app.py：

        count = ingest_documents(...)

    当 return_detail=True 时返回 dict，供 FastAPI 后端使用：

        {
            "parent_chunks": large chunk 数量,
            "child_chunks": small chunk 数量,
            "total_chunks": 总入库 chunk 数量
        }
    """
    large_chunks, small_chunks = split_documents_hierarchical(
        raw_docs,
        source_name,
    )

    docs_to_add = large_chunks + small_chunks

    details = {
        "parent_chunks": len(large_chunks),
        "child_chunks": len(small_chunks),
        "total_chunks": len(docs_to_add),
    }

    if not docs_to_add:
        return details if return_detail else 0

    vector_store.add_documents(docs_to_add)

    if clear_cache:
        clear_cache()
    else:
        _try_clear_streamlit_bm25_cache()

    return details if return_detail else len(docs_to_add)


# ─────────────────────────────────────────────
# 数据源管理
# ─────────────────────────────────────────────

def get_all_sources(vector_store) -> list[str]:
    all_data = vector_store.get()

    sources = sorted(
        {
            meta.get("source", "")
            for meta in all_data.get("metadatas", [])
            if meta and meta.get("source", "") not in _SYSTEM_SOURCES
        }
    )

    return sources


def get_source_info(vector_store, source_name: str) -> dict:
    all_data = vector_store.get()

    metas = [
        meta
        for meta in all_data.get("metadatas", [])
        if meta and meta.get("source") == source_name
    ]

    if not metas:
        return {
            "count": 0,
            "file_type": "unknown",
            "parent_chunks": 0,
            "child_chunks": 0,
        }

    parent_count = sum(
        1
        for m in metas
        if m.get("chunk_type") == "large"
    )

    child_count = sum(
        1
        for m in metas
        if m.get("chunk_type") == "small"
    )

    return {
        "count": len(metas),
        "file_type": metas[0].get("file_type", "unknown"),
        "parent_chunks": parent_count,
        "child_chunks": child_count,
    }


def delete_source_from_db(
    vector_store,
    source_name: str,
    clear_cache: Optional[Callable[[], None]] = None,
) -> int:
    all_data = vector_store.get()

    ids_to_delete = [
        doc_id
        for doc_id, meta in zip(
            all_data.get("ids", []),
            all_data.get("metadatas", []),
        )
        if meta and meta.get("source") == source_name
    ]

    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)

        if clear_cache:
            clear_cache()
        else:
            _try_clear_streamlit_bm25_cache()

    return len(ids_to_delete)


def get_parent_docs_for_children(
    vector_store,
    child_docs: list[Document],
) -> list[Document]:
    """
    根据命中的 small chunk，回溯对应 large parent chunk。

    逻辑：
    1. 检索阶段命中 small chunk。
    2. small chunk 的 metadata 中有 parent_id。
    3. 根据 parent_id 找到 large chunk。
    4. 用 large chunk 作为 LLM 生成答案的上下文。
    """
    parent_ids = []
    parent_score = {}

    for doc in child_docs:
        parent_id = doc.metadata.get("parent_id")

        if not parent_id:
            continue

        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

        score = doc.metadata.get("rerank_score")

        if score is not None:
            parent_score[parent_id] = max(
                float(score),
                float(parent_score.get(parent_id, 0.0)),
            )

    if not parent_ids:
        return child_docs

    all_data = vector_store.get()

    parent_lookup = {}

    for text, meta in zip(
        all_data.get("documents", []),
        all_data.get("metadatas", []),
    ):
        if not meta:
            continue

        chunk_id = meta.get("chunk_id")

        if meta.get("chunk_type") == "large" and chunk_id in parent_ids:
            new_meta = dict(meta)

            if chunk_id in parent_score:
                new_meta["rerank_score"] = parent_score[chunk_id]

            parent_lookup[chunk_id] = Document(
                page_content=text,
                metadata=new_meta,
            )

    parent_docs = [
        parent_lookup[pid]
        for pid in parent_ids
        if pid in parent_lookup
    ]

    return parent_docs or child_docs