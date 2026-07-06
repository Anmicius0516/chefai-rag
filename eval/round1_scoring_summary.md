# ChefAI 正式隔离评估评分总结

## 总体结果

- 总题数：50
- Answer Correctness：0.84 / 1
- Faithfulness：0.47 / 1
- Refusal / Negation Correctness：0.60 / 1

## 分类型结果

| 类型 | 题数 | Answer Correctness | Faithfulness |
|---|---:|---:|---:|
| ingredient | 10 | 0.90 | 0.70 |
| step | 10 | 0.85 | 0.60 |
| detail | 10 | 0.85 | 0.20 |
| compare | 5 | 0.80 | 0.50 |
| follow_up | 10 | 0.90 | 0.30 |
| reject | 3 | 0.67 | 0.67 |
| reject_or_negation | 1 | 1.00 | 1.00 |
| faithfulness_check | 1 | 0.00 | 0.00 |

## 关键问题

1. 基础事实和步骤类问题整体可用，但回答仍然经常加入资料外解释。
2. Menemen 相关检索存在不稳定：q018 中系统把 Menemen 的番茄处理答错，与资料相反。
3. Faithfulness 仍然是最大短板，系统经常把常识、口味建议、传统做法、个人偏好加进去。
4. 多轮追问比连续会话压力测试有所改善，但仍有 q037、q044 这类指代后答案不够准的问题。
5. 拒答仍不稳定：牛排和奶茶能拒答，但宫保鸡丁、冷藏米饭更劲道这类问题会把常识当成答案。

## 建议

下一步不要继续扩功能，优先改 Prompt、拒答规则和 Query Rewrite。完成优化后，用同一套 50 题重跑，形成优化前/优化后对比表。