import pytest

from backend.utils.file_validation import validate_upload_filename


def test_allow_docx():
    assert validate_upload_filename("recipe.docx") == "recipe.docx"


def test_reject_doc():
    with pytest.raises(ValueError):
        validate_upload_filename("old_word.doc")


def test_reject_unknown_extension():
    with pytest.raises(ValueError):
        validate_upload_filename("script.exe")


def test_sanitize_path_name():
    assert validate_upload_filename("../recipe.pdf") == "recipe.pdf"