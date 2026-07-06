from types import SimpleNamespace

from retrieval.confidence import check_retrieval_confidence


def test_no_docs_is_low_confidence():
    assert check_retrieval_confidence([]) == (False, 0.0)


def test_missing_rerank_score_is_low_confidence():
    docs = [SimpleNamespace(metadata={"source": "a.pdf"})]
    assert check_retrieval_confidence(docs) == (False, 0.0)


def test_valid_rerank_score_controls_confidence():
    docs = [SimpleNamespace(metadata={"rerank_score": 0.3})]
    assert check_retrieval_confidence(docs) == (True, 0.3)