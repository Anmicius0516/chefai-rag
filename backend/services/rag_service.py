import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from openai import OpenAI

from config import (
    API_KEY,
    BASE_URL,
    DB_PATH,
    MAX_CONTEXT_DOCS,
    RERANK_URL,
    TOP_K_BM25,
    TOP_K_VECTOR,
    UPLOAD_DIR,
)
from memory.query_rewriter import execute_query_rewriting
from parsers.excel_parser import parse_excel_csv
from parsers.image_parser import parse_image
from parsers.pdf_parser import parse_pdf
from parsers.url_parser import parse_url
from retrieval.confidence import (
    SUPPLEMENT_THRESHOLD,
    check_retrieval_confidence,
    execute_iterative_retrieval,
)
from retrieval.hybrid_search import (
    SimpleRuntimeCache,
    decide_retrieval_strategy,
    execute_hybrid_search,
    execute_rerank,
    execute_search_by_strategy,
)
from vectorstore.chroma_manager import (
    ZhipuEmbeddingService,
    delete_source_from_db,
    get_all_sources,
    get_parent_docs_for_children,
    get_source_info,
    ingest_documents,
    init_vector_store,
)


load_dotenv()


class ChefAIRAGService:
    def __init__(self):
        if not API_KEY:
            raise RuntimeError("缺少 ZHIPU_API_KEY，请先检查 .env 配置")

        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.rerank_url = RERANK_URL
        self.db_path = DB_PATH

        self.upload_dir = Path(UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.embedding_service = ZhipuEmbeddingService(self.client)
        self.vector_store = init_vector_store(DB_PATH, self.embedding_service)

        self.runtime_cache = SimpleRuntimeCache()

        self.sessions: Dict[str, List[dict]] = {}
        self.session_summaries: Dict[str, str] = {}

    def parse_local_file(self, file_path: str, source_name: str) -> List[Document]:
        ext = os.path.splitext(source_name)[-1].lower()

        if ext == ".pdf":
            return parse_pdf(file_path, source_name, self.client)

        if ext in [".xlsx", ".xls", ".csv"]:
            return parse_excel_csv(file_path, source_name)

        if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            return parse_image(file_path, source_name, self.client)

        if ext == ".docx":
            docs = Docx2txtLoader(file_path).load()
            for doc in docs:
                doc.metadata["source"] = source_name
                doc.metadata["file_type"] = "docx"
            return docs

        if ext in [".txt", ".md"]:
            docs = TextLoader(file_path, encoding="utf-8").load()
            for doc in docs:
                doc.metadata["source"] = source_name
                doc.metadata["file_type"] = ext.replace(".", "")
            return docs

        raise ValueError(f"不支持的文件类型：{ext}")

    def ingest_file(self, file_path: str, source_name: str) -> dict:
        raw_docs = self.parse_local_file(file_path, source_name)

        result = ingest_documents(
            self.vector_store,
            raw_docs,
            source_name,
            clear_cache=self.runtime_cache.clear_bm25_cache,
            return_detail=True,
        )

        return result

    def ingest_url(self, url: str) -> dict:
        docs, err = parse_url(url)

        if err:
            raise RuntimeError(err)

        result = ingest_documents(
            self.vector_store,
            docs,
            url,
            clear_cache=self.runtime_cache.clear_bm25_cache,
            return_detail=True,
        )

        return result

    def get_messages(self, session_id: str) -> List[dict]:
        return self.sessions.setdefault(session_id, [])

    def compress_history_to_summary(
        self,
        session_id: str,
        messages: List[dict],
        history_compress_rounds: int = 8,
    ) -> str:
        if len(messages) < history_compress_rounds:
            return self.session_summaries.get(session_id, "")

        old_messages = messages[:-4]
        history_text = "\n".join(
            f"{m['role']}: {m['content'][:300]}"
            for m in old_messages
        )

        prompt = (
            "请将以下烹饪对话历史压缩为150字以内的摘要，保留：\n"
            "1. 用户提过的食材偏好和饮食限制\n"
            "2. 已推荐过的食谱名称\n"
            "3. 用户的口味偏好（辣度、清淡等）\n\n"
            f"【对话历史】\n{history_text}\n\n"
            "请输出摘要："
        )

        try:
            resp = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )

            summary = resp.choices[0].message.content.strip()
            self.session_summaries[session_id] = summary
            return summary

        except Exception:
            return self.session_summaries.get(session_id, "")

    def get_effective_history(
        self,
        session_id: str,
        messages: List[dict],
    ) -> Tuple[List[dict], str]:
        recent = messages[-4:] if len(messages) > 4 else messages
        summary = self.compress_history_to_summary(session_id, messages)
        return recent, summary

    def run_retrieval(self, question: str, session_id: str):
        messages = self.get_messages(session_id)
        strategy = decide_retrieval_strategy(question, self.client)

        rewritten_query = execute_query_rewriting(
            question,
            messages,
            self.client,
        )

        if strategy == "skip":
            return [], False, 0.0, rewritten_query, strategy

        candidate_docs = execute_search_by_strategy(
            rewritten_query,
            self.vector_store,
            self.runtime_cache,
            strategy=strategy,
            k_vector=TOP_K_VECTOR,
            k_bm25=TOP_K_BM25,
        )

        if not candidate_docs:
            return [], False, 0.0, rewritten_query, strategy

        reranked_docs = execute_rerank(
            rewritten_query,
            candidate_docs,
            self.api_key,
            self.rerank_url,
            top_n=MAX_CONTEXT_DOCS,
        )

        final_docs = execute_iterative_retrieval(
            rewritten_query,
            reranked_docs,
            self.client,
            lambda q, k_vector, k_bm25: execute_hybrid_search(
                q,
                self.vector_store,
                self.runtime_cache,
                k_vector,
                k_bm25,
            ),
            lambda q, docs, top_n: execute_rerank(
                q,
                docs,
                self.api_key,
                self.rerank_url,
                top_n,
            ),
        )

        is_confident, score = check_retrieval_confidence(final_docs)

        return final_docs, is_confident, score, rewritten_query, strategy

    def build_context_docs(self, retrieved_docs: List[Document]) -> List[Document]:
        parent_docs = get_parent_docs_for_children(
            self.vector_store,
            retrieved_docs,
        )

        return parent_docs[:MAX_CONTEXT_DOCS]

    def generate_answer(
        self,
        question: str,
        session_id: str,
        context_docs: List[Document],
    ) -> str:
        """
        严格基于检索上下文生成答案。

        关键调整：
        1. 生成阶段只使用 rewritten question + retrieved context，不再拼接历史对话。
           多轮问题由 Query Rewrite 负责消解，避免历史回答污染生成。
        2. temperature=0，降低自由发挥。
        3. Prompt 明确禁止常识扩展、传统做法、个人建议和相近含义推断。
        """
        context_payload = "\n\n".join(
            f"[来源 {i + 1}] {doc.metadata.get('source', '未知来源')}\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        )

        system_prompt = f"""
你是 ChefAI，一个严格基于知识库回答问题的中文助手。你的目标不是“回答得丰富”，而是“回答得可验证”。

【可用资料】
{context_payload}

【最高优先级规则】
1. 只能根据【可用资料】回答；资料没有逐字或明确语义支持的内容，一律不要写。
2. 禁止使用以下表达：通常、一般、传统上、可以理解为、暗示、可能、个人口味、更加美味、营养、口感更好、风味更丰富。
3. 如果资料不足，直接回答：“当前知识库中未提供相关内容。”
4. 如果问题里有“有没有说 / 有没有提到 / 根据资料 / 是否必须 / 一定要”，只判断资料是否【明确】写了；不能用推断、暗示、相近意思回答。
5. 如果资料没有明确写“更劲道”，就必须回答“资料没有明确提到更劲道”，不能说“可以理解为”。
6. 如果问题中的菜名或实体没有出现在资料里，必须回答“当前知识库中未提供相关内容”，不能补充常识或传统做法。
7. 如果问题只涉及某些菜品，忽略与这些菜品无关的资料；例如问“番茄炒蛋和 Menemen”，不要使用炒饭资料。
8. 回答尽量短：先给结论，再列出资料中明确支持的要点。不要扩写。
9. 输出时不要给替代做法、不要给小贴士、不要评价味道，除非资料明确写了。

【回答格式】
- 普通问题：直接回答资料中明确写出的内容。
- “有没有说/根据资料/是否必须”类问题：先回答“有/没有明确提到”，再说明资料原本说了什么。
""".strip()

        messages_payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        resp = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=messages_payload,
            temperature=0,
            max_tokens=500,
        )

        return resp.choices[0].message.content.strip()

    def chat(self, question: str, session_id: str = "default") -> dict:
        (
            retrieved_docs,
            is_confident,
            score,
            rewritten_query,
            strategy,
        ) = self.run_retrieval(question, session_id)

        if strategy == "skip":
            answer = (
                "你好，我是 ChefAI。你可以上传食谱、表格、图片或网页 URL，"
                "然后问我具体做法、食材替换、步骤总结等问题。"
            )
            context_docs = []
            is_confident = False
            score = 0.0

        elif not retrieved_docs:
            answer = (
                "我没有在当前知识库中检索到相关内容。"
                "你可以先上传食谱文档、表格、图片或网页 URL 后再提问。"
            )
            context_docs = []
            is_confident = False
            score = 0.0

        elif not is_confident and score < SUPPLEMENT_THRESHOLD:
            answer = (
                "当前知识库中相关资料不足，我不能确定答案。"
                "建议补充上传更直接相关的食谱，或换一个更具体的问题。"
            )
            context_docs = retrieved_docs[:MAX_CONTEXT_DOCS]

        else:
            context_docs = self.build_context_docs(retrieved_docs)
            # 使用改写后的独立查询生成答案，避免“它/这个”等指代问题依赖长历史。
            answer = self.generate_answer(rewritten_query, session_id, context_docs)

        messages = self.get_messages(session_id)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "session_id": session_id,
            "is_confident": bool(is_confident),
            "confidence_score": float(score or 0.0),
            "rewritten_query": rewritten_query,
            "retrieval_strategy": strategy,
            "sources": self.format_sources(context_docs if retrieved_docs else []),
        }

    def format_sources(self, docs: List[Document]) -> List[dict]:
        items = []
        seen = set()

        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            key = (source, page, doc.page_content[:60])

            if key in seen:
                continue

            seen.add(key)

            items.append(
                {
                    "source": source,
                    "page": page,
                    "file_type": doc.metadata.get("file_type"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "content_preview": doc.page_content[:260].replace("\n", " "),
                }
            )

        return items

    def list_sources(self) -> List[dict]:
        result = []

        for source in get_all_sources(self.vector_store):
            info = get_source_info(self.vector_store, source)
            result.append(
                {
                    "source": source,
                    **info,
                }
            )

        return result

    def delete_source(self, source_name: str) -> int:
        return delete_source_from_db(
            self.vector_store,
            source_name,
            clear_cache=self.runtime_cache.clear_bm25_cache,
        )

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)
        self.session_summaries.pop(session_id, None)