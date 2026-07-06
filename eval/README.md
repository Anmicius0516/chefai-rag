# RAG 评估文件说明

- `eval_dataset.jsonl`：50 条中文 QA 评估集。
- `round1_formal_eval_results.csv`：优化前正式评估结果。
- `round2_formal_eval_results.csv`：优化后正式评估结果。
- `round1_scoring_summary.md` / `round2_scoring_summary.md`：两轮评分总结。

评分指标：

- Answer Correctness：回答是否正确。
- Faithfulness：是否忠实于检索资料，是否避免资料外扩展。
- Refusal / Negation Correctness：知识库外问题和否定类问题是否能正确拒答或严格判断。
