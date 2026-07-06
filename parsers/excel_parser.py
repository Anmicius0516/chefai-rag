"""
excel_parser.py

负责：
1. Excel解析
2. CSV解析
3. 表格语义化
"""

import os

import pandas as pd

from langchain_core.documents import Document


def parse_excel_csv(
    file_path: str,
    source_name: str,
):
    """
    Excel / CSV解析
    """

    ext = os.path.splitext(
        file_path
    )[-1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    df = df.fillna("").astype(str)

    raw_docs = []

    overview = (
        f"食谱数据表来源：{source_name}\n"
        f"共 {len(df)} 行\n"
        f"共 {len(df.columns)} 列\n"
        f"列名：{', '.join(df.columns.tolist())}"
    )

    raw_docs.append(
        Document(
            page_content=overview,
            metadata={
                "source": source_name,
                "page": 0,
                "file_type": "table",
                "row_index": -1
            }
        )
    )

    for idx, row in df.iterrows():

        parts = [
            f"{col}为“{val}”"
            for col, val in row.items()
            if str(val).strip()
        ]

        if not parts:
            continue

        text = (
            f"[第{idx+1}条] "
            + "，".join(parts)
            + "。"
        )

        raw_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "page": idx + 1,
                    "file_type": "table",
                    "row_index": idx
                }
            )
        )

    return raw_docs