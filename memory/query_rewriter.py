"""
memory/query_rewriter.py

纯 Python Query Rewrite 模块。
不能 import Streamlit，保证 FastAPI 后端可独立运行。
"""

from typing import List


def execute_query_rewriting(
    user_question: str,
    chat_history: List[dict],
    llm_client,
) -> str:
    """
    多轮指代消解。

    作用：
    把“这个怎么做”“还能换成鸡胸肉吗”这类依赖上下文的问题，
    改写成可以独立检索的查询。
    """
    question = (user_question or "").strip()
    if not question:
        return user_question

    if not chat_history:
        return question

    # 只有明显依赖上下文的短指代问题才改写。
    # 明确问题不要反复调用 LLM 改写，避免把清晰问题改偏。
    context_dependent_markers = [
        "它", "这个", "这道", "这种", "前面", "刚才", "上面",
        "该", "其中", "这个步骤", "这个配料", "这两个菜",
    ]
    if not any(marker in question for marker in context_dependent_markers):
        return question

    compact_history = []
    for m in chat_history[-2:]:
        content = str(m.get("content", "")).replace("\n", " ").strip()
        if not content:
            continue
        limit = 180 if m.get("role") == "user" else 120
        compact_history.append(f"{m.get('role', 'unknown')}: {content[:limit]}")

    history_context = "\n".join(compact_history)

    if not history_context.strip():
        return question

    prompt = (
        "你是一个 RAG 检索查询改写器，只负责指代消解。\n"
        "请结合最近一轮上下文，把当前输入改写成一个可以独立检索的中文查询。\n"
        "要求：\n"
        "1. 只改写查询，不要回答问题。\n"
        "2. 必须优先使用最近一轮用户问题中的菜名或食材。\n"
        "3. 如果当前问题中的‘它/这个/这道菜/这个配料’指代不清，只选择最近一轮上下文的对象。\n"
        "4. 不要引入更早对话里的菜名。\n"
        "5. 不要扩展成做法、建议或解释。\n"
        "6. 只输出改写后的查询本身。\n\n"
        f"【最近上下文】\n{history_context}\n\n"
        f"【当前输入】\n{question}\n\n"
        "改写后的查询："
    )

    try:
        resp = llm_client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten or question
    except Exception:
        return question