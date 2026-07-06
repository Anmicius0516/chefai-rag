# ChefAI 第2轮正式评估评分总结

## 总体结果

- 总题数：50
- Answer Correctness：0.89 / 1
- Faithfulness：0.91 / 1
- Refusal / Negation Correctness：1.00 / 1

## 与第1轮对比

| 指标 | 第1轮正式评估 | 第2轮正式评估 | 变化 |
|---|---:|---:|---:|
| Answer Correctness | 0.84 | 0.89 | +0.05 |
| Faithfulness | 0.47 | 0.91 | +0.44 |
| Refusal / Negation | 0.60 | 1.00 | +0.40 |

## 分类型结果

| 类型 | 题数 | Answer Correctness | Faithfulness |
|---|---:|---:|---:|
| ingredient | 10 | 0.90 | 1.00 |
| step | 10 | 0.80 | 1.00 |
| detail | 10 | 0.95 | 0.90 |
| compare | 5 | 0.70 | 0.60 |
| follow_up | 10 | 0.95 | 0.85 |
| reject | 3 | 1.00 | 1.00 |
| reject_or_negation | 1 | 1.00 | 1.00 |
| faithfulness_check | 1 | 1.00 | 1.00 |

## 关键结论

1. 第2轮优化有效，Faithfulness 和拒答能力明显提升。
2. 当前仍有局部问题：糖的作用回答不准、Menemen 主要食材遗漏、相同点/区别类问题仍会做轻微资料外扩展。
3. “番茄炒蛋怎么做”与“番茄炒蛋这道菜怎么做”表现不同，属于 query routing / rewrite / 检索触发问题，建议作为后续优化点，但不建议在本轮继续大改。

## 简历可用表述

构建50条中文QA评估集，覆盖食材、步骤、细节、相似菜品对比、多轮追问和知识库外拒答；基于首轮评估定位Faithfulness和拒答不足，优化生成Prompt、拒答规则和Query Rewrite策略，使Faithfulness由0.47提升至0.91，拒答/否定类正确率由0.60提升至1.00。