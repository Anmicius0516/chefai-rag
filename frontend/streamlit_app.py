import os
import uuid
from typing import Any, Dict, Optional

import requests
import streamlit as st
from dotenv import load_dotenv


# =========================
# 1. 基础配置
# =========================

load_dotenv()

API_BASE_URL = os.getenv("CHEFAI_API_URL", "http://127.0.0.1:8000").rstrip("/")

# 关键修复：
# 禁止 requests 读取系统代理环境变量，避免 127.0.0.1 请求被代理拦截导致 HTTP 503。
http_session = requests.Session()
http_session.trust_env = False


st.set_page_config(
    page_title="ChefAI 智能烹饪问答系统",
    page_icon="🍳",
    layout="wide",
)


# =========================
# 2. Session 初始化
# =========================

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = False


# =========================
# 3. 通用工具函数
# =========================

def safe_rerun():
    """
    兼容不同 Streamlit 版本的刷新方法。
    """
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def format_response_error(res: requests.Response) -> str:
    """
    把后端返回的错误整理成可读文本。
    避免只显示一个空的粉色/黄色错误框。
    """
    try:
        data = res.json()
        detail = data.get("detail", data)
    except Exception:
        detail = res.text

    return f"HTTP {res.status_code}\n{detail}"


def api_get(path: str, timeout: int = 30) -> requests.Response:
    """
    GET 请求 FastAPI 后端。
    使用 http_session，避免走系统代理。
    """
    return http_session.get(
        f"{API_BASE_URL}{path}",
        timeout=timeout,
    )


def api_post(path: str, timeout: int = 180, **kwargs) -> requests.Response:
    """
    POST 请求 FastAPI 后端。
    使用 http_session，避免走系统代理。
    """
    return http_session.post(
        f"{API_BASE_URL}{path}",
        timeout=timeout,
        **kwargs,
    )


def api_delete(path: str, timeout: int = 60, **kwargs) -> requests.Response:
    """
    DELETE 请求 FastAPI 后端。
    使用 http_session，避免走系统代理。
    """
    return http_session.delete(
        f"{API_BASE_URL}{path}",
        timeout=timeout,
        **kwargs,
    )


def check_backend(show_message: bool = True) -> bool:
    """
    检查 FastAPI 后端是否正常。
    """
    try:
        res = api_get("/health", timeout=10)

        if res.ok:
            data = res.json()
            st.session_state.backend_ok = True

            if show_message:
                st.success("后端连接正常")
                st.write(f"状态：{data.get('status')}")
                st.write(f"向量库路径：{data.get('vector_db_path')}")
                st.write(f"API Key 状态：{data.get('has_api_key')}")

            return True

        st.session_state.backend_ok = False

        if show_message:
            st.error("后端连接失败")
            st.code(format_response_error(res))

        return False

    except Exception as e:
        st.session_state.backend_ok = False

        if show_message:
            st.error("无法连接 FastAPI 后端")
            st.code(str(e))

        return False


def load_sources() -> Optional[list]:
    """
    加载知识源列表。

    返回：
    - list：正常返回
    - None：请求失败
    """
    try:
        res = api_get("/api/sources", timeout=30)

        if not res.ok:
            st.warning("知识源列表读取失败")
            st.code(format_response_error(res))
            return None

        data = res.json()
        return data.get("sources", [])

    except Exception as e:
        st.warning("知识源列表读取失败")
        st.code(str(e))
        return None


# =========================
# 4. 页面标题
# =========================

st.title("🍳 ChefAI · RAG 智能烹饪问答系统")
st.caption("FastAPI 后端 + Streamlit 前端：文档入库、混合检索、Rerank、多轮问答")


# =========================
# 5. 侧边栏：知识库管理
# =========================

with st.sidebar:
    st.header("📚 知识库管理")
    st.caption(f"后端地址：{API_BASE_URL}")

    if st.button("检查后端连接"):
        check_backend(show_message=True)

    st.divider()

    # ---------- 文件上传 ----------
    st.subheader("上传食谱文件")

    uploaded_file = st.file_uploader(
        "选择文件",
        type=[
            "txt",
            "pdf",
            "docx",
            "doc",
            "md",
            "xlsx",
            "xls",
            "csv",
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ],
        help="建议先用 txt 测试，确认流程跑通后再测试 PDF、图片、Excel。",
    )

    if uploaded_file is not None:
        st.info(
            f"已选择文件：{uploaded_file.name}\n\n"
            f"文件大小：{uploaded_file.size / 1024:.2f} KB"
        )

    if uploaded_file is not None and st.button("解析并入库", type="primary"):
        try:
            with st.spinner("正在上传、解析并写入知识库..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }

                res = api_post(
                    "/api/upload",
                    files=files,
                    timeout=300,
                )

            if res.ok:
                data = res.json()

                st.success(data.get("message", "文件入库成功"))
                st.write(f"知识源：{data.get('source_name')}")
                st.write(f"Parent chunks：{data.get('parent_chunks')}")
                st.write(f"Child chunks：{data.get('child_chunks')}")
                st.write(f"Total chunks：{data.get('total_chunks')}")

                safe_rerun()

            else:
                st.error("文件入库失败")
                st.code(format_response_error(res))

        except Exception as e:
            st.error("上传或入库过程出错")
            st.code(str(e))

    st.divider()

    # ---------- URL 导入 ----------
    st.subheader("导入网页 URL")

    url_input = st.text_input(
        "网页地址",
        placeholder="例如：https://example.com/recipe",
    )

    if url_input and st.button("抓取网页并入库"):
        try:
            with st.spinner("正在抓取网页并写入知识库..."):
                res = api_post(
                    "/api/url",
                    json={"url": url_input},
                    timeout=300,
                )

            if res.ok:
                data = res.json()

                st.success(data.get("message", "网页入库成功"))
                st.write(f"知识源：{data.get('source_name')}")
                st.write(f"Parent chunks：{data.get('parent_chunks')}")
                st.write(f"Child chunks：{data.get('child_chunks')}")
                st.write(f"Total chunks：{data.get('total_chunks')}")

                safe_rerun()

            else:
                st.error("网页入库失败")
                st.code(format_response_error(res))

        except Exception as e:
            st.error("网页导入过程出错")
            st.code(str(e))

    st.divider()

    # ---------- 知识源列表 ----------
    st.subheader("已有知识源")

    sources = load_sources()

    if sources is None:
        st.warning("知识源列表读取失败，请先确认后端是否正常运行。")

    elif len(sources) == 0:
        st.info("暂无知识源，请先上传 txt / PDF / 图片 / 表格 / URL。")

    else:
        for item in sources:
            source_name = item.get("source", "未知来源")

            with st.expander(source_name):
                st.write(f"文件类型：{item.get('file_type')}")
                st.write(f"总 chunks：{item.get('count')}")
                st.write(f"Parent chunks：{item.get('parent_chunks')}")
                st.write(f"Child chunks：{item.get('child_chunks')}")

                if st.button(
                    "删除该知识源",
                    key=f"delete_{source_name}",
                ):
                    try:
                        del_res = api_delete(
                            "/api/sources",
                            params={"source": source_name},
                            timeout=60,
                        )

                        if del_res.ok:
                            data = del_res.json()
                            st.success(data.get("message", "删除完成"))
                            st.write(f"删除 chunks 数量：{data.get('deleted_chunks')}")
                            safe_rerun()
                        else:
                            st.error("删除失败")
                            st.code(format_response_error(del_res))

                    except Exception as e:
                        st.error("删除过程出错")
                        st.code(str(e))

    st.divider()

    # ---------- 会话管理 ----------
    st.subheader("会话管理")

    st.caption(f"当前 session_id：{st.session_state.session_id}")

    if st.button("清空当前前端会话"):
        try:
            api_delete(f"/api/session/{st.session_state.session_id}", timeout=30)
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.success("当前会话已清空")
        safe_rerun()


# =========================
# 6. 主聊天区域
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("请输入你的问题，例如：红烧肉怎么做？")


if question:
    # 用户消息写入前端会话
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                res = api_post(
                    "/api/chat",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=300,
                )

                if not res.ok:
                    error_text = format_response_error(res)
                    st.error("问答接口请求失败")
                    st.code(error_text)

                    answer = f"请求失败：\n\n```text\n{error_text}\n```"

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                else:
                    data: Dict[str, Any] = res.json()

                    answer = data.get("answer", "")
                    st.markdown(answer)

                    is_confident = data.get("is_confident")
                    confidence_score = data.get("confidence_score", 0.0)
                    rewritten_query = data.get("rewritten_query", "")

                    st.caption(
                        f"检索置信度：{is_confident} | "
                        f"score={confidence_score:.4f} | "
                        f"改写查询：{rewritten_query}"
                    )

                    sources = data.get("sources", [])

                    if sources:
                        with st.expander("查看引用来源"):
                            for i, src in enumerate(sources, 1):
                                st.markdown(
                                    f"**来源 {i}: {src.get('source', '未知来源')}**"
                                )

                                if src.get("page") is not None:
                                    st.write(f"页码/位置：{src.get('page')}")

                                if src.get("file_type"):
                                    st.write(f"文件类型：{src.get('file_type')}")

                                if src.get("rerank_score") is not None:
                                    st.write(f"Rerank Score：{src.get('rerank_score')}")

                                st.write(src.get("content_preview", ""))
                                st.divider()

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

            except Exception as e:
                error_msg = f"请求失败：{e}"
                st.error(error_msg)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )