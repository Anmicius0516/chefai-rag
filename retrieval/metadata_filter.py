"""
metadata_filter.py

负责：
1. 食谱元数据抽取
2. Metadata过滤
3. Cuisine / Method / Diet 标签管理
"""

from typing import Dict, List
from langchain_core.documents import Document


def extract_metadata_tags(
    text: str,
    source: str,
    page: int
) -> Dict:
    """
    从食谱文本抽取结构化标签
    """

    # 菜系识别
    cuisine_map = {
        "川菜": "川菜",
        "粤菜": "粤菜",
        "湘菜": "湘菜",
        "淮扬": "淮扬菜",
        "意大利": "意式",
        "法式": "法式",
        "日式": "日式",
        "韩式": "韩式",
        "泰式": "泰式",
        "墨西哥": "墨西哥",
        "中式": "中式",
        "西式": "西式",
    }

    cuisine = next(
        (v for k, v in cuisine_map.items() if k in text),
        "其他"
    )

    # 烹饪方式识别
    method_map = {
        "炒": "炒",
        "蒸": "蒸",
        "炸": "炸",
        "烤": "烤",
        "炖": "炖",
        "煮": "煮",
        "焖": "焖",
        "拌": "凉拌",
        "煎": "煎",
        "烩": "烩",
    }

    method = next(
        (v for k, v in method_map.items() if k in text),
        "其他"
    )

    # 饮食标签
    diet_tags = []

    if any(
        kw in text
        for kw in ["无麸质", "gluten-free", "不含面粉"]
    ):
        diet_tags.append("无麸质")

    if any(
        kw in text
        for kw in ["素食", "纯素", "vegan", "vegetarian"]
    ):
        diet_tags.append("素食")

    if any(
        kw in text
        for kw in ["低卡", "减脂", "低热量"]
    ):
        diet_tags.append("低卡")

    if any(
        kw in text
        for kw in ["辣", "麻辣", "香辣"]
    ):
        diet_tags.append("辣")

    return {
        "source": source,
        "page": page,
        "cuisine": cuisine,
        "method": method,
        "diet_tags": ",".join(diet_tags)
        if diet_tags else "普通",
        "chunk_type": "text",
    }


# ==========================
# Metadata Filters
# ==========================

def filter_by_cuisine(
    docs: List[Document],
    cuisine: str
) -> List[Document]:

    return [
        doc
        for doc in docs
        if doc.metadata.get("cuisine") == cuisine
    ]


def filter_by_method(
    docs: List[Document],
    method: str
) -> List[Document]:

    return [
        doc
        for doc in docs
        if doc.metadata.get("method") == method
    ]


def filter_by_diet(
    docs: List[Document],
    diet: str
) -> List[Document]:

    return [
        doc
        for doc in docs
        if diet in doc.metadata.get("diet_tags", "")
    ]


def apply_metadata_filters(
    docs: List[Document],
    cuisine: str = None,
    method: str = None,
    diet: str = None,
) -> List[Document]:
    """
    统一Metadata过滤入口
    """

    results = docs

    if cuisine:
        results = filter_by_cuisine(results, cuisine)

    if method:
        results = filter_by_method(results, method)

    if diet:
        results = filter_by_diet(results, diet)

    return results