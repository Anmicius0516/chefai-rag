"""
memory/conversation_memory.py

纯 Python 会话记忆工具。
注意：本文件不能 import Streamlit。
FastAPI 后端、CLI 脚本、测试环境都可以复用。
"""

from typing import Callable, List, Optional, Tuple


def compress_history_to_summary(
    messages: List[dict],
    llm_client,
    previous_summary: str = "",
    history_compress_rounds: int = 8,
    on_summary_update: Optional[Callable[[str], None]] = None,
) -> str:
    """
    当历史超过指定轮数时，将前段历史压缩为摘要。

    previous_summary:
        外部传入的旧摘要，例如 FastAPI service 中的 session_summaries[session_id]。
    on_summary_update:
        摘要生成成功后的回调，调用方可用它写回 Redis / dict / DB。
    """
    if len(messages) < history_compress_rounds:
        return previous_summary or ""

    old_messages = messages[:-4]
    history_text = "\n".join(
        f"{m.get('role', 'unknown')}: {str(m.get('content', ''))[:300]}"
        for m in old_messages
        if m.get("content")
    )

    if not history_text.strip():
        return previous_summary or ""

    prompt = (
        "请将以下烹饪对话历史压缩为150字以内的摘要，保留：\n"
        "1. 用户提过的食材偏好和饮食限制\n"
        "2. 已推荐过的食谱名称\n"
        "3. 用户的口味偏好（辣度、清淡等）\n\n"
        f"【对话历史】\n{history_text}\n\n"
        "请输出摘要："
    )

    try:
        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        summary = resp.choices[0].message.content.strip()
        if summary and on_summary_update:
            on_summary_update(summary)
        return summary or previous_summary or ""
    except Exception:
        return previous_summary or ""


def get_effective_history(
    messages: List[dict],
    llm_client,
    previous_summary: str = "",
    history_compress_rounds: int = 8,
    on_summary_update: Optional[Callable[[str], None]] = None,
) -> Tuple[List[dict], str]:
    recent = messages[-4:] if len(messages) > 4 else messages
    summary = compress_history_to_summary(
        messages=messages,
        llm_client=llm_client,
        previous_summary=previous_summary,
        history_compress_rounds=history_compress_rounds,
        on_summary_update=on_summary_update,
    )
    return recent, summary