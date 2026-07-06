"""
image_parser.py

负责：
1. 菜谱图片 OCR
2. 菜品视觉理解
3. 图片转 Document

核心修复：
- 图片识别失败时直接抛异常。
- 不再把“图片识别失败”这种错误文本写进知识库。
"""

import base64
import os

from langchain_core.documents import Document


IMAGE_MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "bmp": "image/bmp",
}


def _looks_like_failed_vision_result(content: str) -> bool:
    if not content or len(content.strip()) < 8:
        return True

    failure_keywords = [
        "无法识别",
        "不能识别",
        "识别失败",
        "无法读取",
        "看不清",
        "抱歉",
        "sorry",
        "cannot",
        "can't",
        "unable",
    ]
    lower_content = content.lower()
    return any(keyword.lower() in lower_content for keyword in failure_keywords)


def parse_image(
    file_path: str,
    source_name: str,
    client,
):
    ext = os.path.splitext(file_path)[-1].lower().replace(".", "")
    mime = IMAGE_MIME_MAP.get(ext)

    if not mime:
        raise ValueError(f"不支持的图片格式：.{ext}")

    try:
        with open(file_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as exc:
        raise RuntimeError(f"图片读取失败：{exc}") from exc

    try:
        resp = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{img_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "请完整分析该菜谱图片：\n"
                                "1. 识别全部文字\n"
                                "2. 提取食材和用量\n"
                                "3. 提取烹饪步骤\n"
                                "4. 判断菜系和口味\n"
                                "5. 描述最终成品\n"
                                "如果图片不是菜谱或无法识别，请明确说明无法识别。"
                            ),
                        },
                    ],
                }
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"图片识别失败，不入库：{exc}") from exc

    if _looks_like_failed_vision_result(content):
        raise RuntimeError("图片识别结果为空或不可靠，不入库")

    return [
        Document(
            page_content=f"[图像食谱]\n{content}",
            metadata={
                "source": source_name,
                "page": 1,
                "file_type": "image",
            },
        )
    ]