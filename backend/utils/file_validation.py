"""
backend/utils/file_validation.py

统一管理上传文件白名单。
后端先校验后保存，避免未知格式进入解析链路。
"""

from pathlib import Path


ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".xlsx",
    ".xls",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

ALLOWED_UPLOAD_EXTENSIONS_TEXT = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))


def get_file_extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def sanitize_filename(filename: str | None) -> str:
    safe_name = Path(filename or "uploaded_file").name.strip()
    return safe_name or "uploaded_file"


def validate_upload_filename(filename: str | None) -> str:
    safe_name = sanitize_filename(filename)
    ext = get_file_extension(safe_name)

    if not ext:
        raise ValueError("文件缺少扩展名，无法判断类型")

    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(
            "不支持的文件类型："
            f"{ext}。当前仅支持：{ALLOWED_UPLOAD_EXTENSIONS_TEXT}"
        )

    return safe_name