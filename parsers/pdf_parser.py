"""
pdf_parser.py

功能：
1. 解析 PDF 文本
2. 支持图片页 OCR
3. 避免 silent fail

修复：
- PDF 图片页 OCR 失败时不再写入“[OCR失败]”到知识库。
- 如果整个 PDF 没有任何有效内容，直接抛异常。
"""

import base64
import io

import pdfplumber
from langchain_core.documents import Document


def ocr_page_via_vision(page, client) -> str:
    try:
        pil_img = page.to_image(resolution=150).original

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {
                            "type": "text",
                            "text": "请提取图片中的文字，保持结构化输出。如果无法识别，请说明无法识别。",
                        },
                    ],
                }
            ],
            temperature=0,
        )

        content = resp.choices[0].message.content.strip()
        if not content or "无法识别" in content or "识别失败" in content:
            raise RuntimeError("OCR 结果为空或不可靠")
        return content

    except Exception as e:
        raise RuntimeError(f"OCR识别失败: {e}") from e


def parse_pdf(file_path: str, source_name: str, client):
    raw_docs = []
    ocr_errors = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                has_images = len(page.images) > 0

                if text.strip():
                    raw_docs.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "source": source_name,
                                "page": i + 1,
                                "has_image": has_images,
                                "file_type": "pdf",
                            },
                        )
                    )
                    continue

                if has_images:
                    try:
                        img_text = ocr_page_via_vision(page, client)
                        raw_docs.append(
                            Document(
                                page_content=f"[OCR图像内容]\n{img_text}",
                                metadata={
                                    "source": source_name,
                                    "page": i + 1,
                                    "has_image": True,
                                    "file_type": "pdf_image",
                                },
                            )
                        )
                    except Exception as e:
                        ocr_errors.append(f"第 {i + 1} 页 OCR 失败：{e}")

        if not raw_docs:
            if ocr_errors:
                raise RuntimeError("PDF解析失败：未提取到有效内容；" + "；".join(ocr_errors[:3]))
            raise RuntimeError("PDF解析失败：未提取到任何内容")

        return raw_docs

    except Exception as e:
        raise RuntimeError(f"PDF解析整体失败: {e}") from e