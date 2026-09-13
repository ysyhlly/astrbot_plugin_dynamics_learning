# Dynamics Learning ↔ ChatDynamics 只读契约

本插件是 ChatDynamics 的附属插件，对本体的所有数据只读。它没有、也不会有任何写入本体
配置或 KV 的代码路径。这份文档说明它到底读什么、依赖哪些字段，以及本体升级时怎么判断
是否破坏契约。

## 1. 读取方式

通过 AstrBot 的共享首选项按作用域整体读取：

```python
from astrbot.core import sp
rows = await sp.range_get_async("plugin", source_plugin_id, None)
```

`source_plugin_id` 默认 `ysyhlly/astrbot_plugin_chat_dynamics`，与
`astrbot.core.star.star.StarMetadata.plugin_id` 的 `{author}/{name}` 规则一致
（两侧都小写）。本体更换作者或插件名时必须同步修改本插件配置里的
`source_plugin_id`，否则导入会返回 `unknown_sessions` 而不是静默出错。

不使用 HTTP、不导入本体的 Python 模块、不读取本体的数据目录。本体重载或未安装时，
本插件只是导入到 0 条并给出诊断，不影响机器人运行。

## 2. 依赖的两类键

| 键 | 写入方（ChatDynamics 侧） | 本插件用途 |
| --- | --- | --- |
| `panel_runtime_v1` | `core/runtime_persistence.export_runtime_state` | 会话列表；把标注键的哈希还原成 session_key |
| `topic_annotations_v1_<sha256(session)>` | `core/topic_annotations.TopicAnnotations.save` | 人工标签 + 冻结的 `decision_trace` 快照 |

标注键只存哈希，所以 session 名要从 `panel_runtime_v1.sessions[].session_key`
反推。**无法归属的标注会被统计进 `diagnostics.unknown_sessions` 并跳过**：话题指标只在
会话内比较，没有会话的标注无法参与评分，猜测归属会比丢掉它更糟。

## 3. 标注记录里用到的字段

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `msg_id` | str | 样本主键的一半；必须非空 |
| `annotated_at` | float | 同一消息重复标注时取最新 |
| `predicted_topic` | str | 话题任务的预测；`UNKNOWN` 表示未归属 |
| `expected_topic` | str | 话题任务的标注；`NEW` 按单例处理，`UNKNOWN` 不计分 |
| `error_type` | str | 话题错误类型统计 |
| `bot_targeted` | bool | 收件人任务标签；缺失则不产生收件人样本 |
| `recipient_error_type` | str | 收件人错误类型统计 |
| `expected_reply` | bool | 回复准入任务标签 |
| `routing.topic_confidence` / `routing.topic_ambiguous` | float/bool | 话题阈值回放 |
| `routing.topic_candidates` | list | **放宽**归属阈值的回放；缺失时该方向不可回放 |
| `decision_trace` | dict | 证据族、贡献值、身份与状态特征 |

字段缺失一律降级，不抛异常：`core/trace.py` 的 `parse_decision_trace` 对任何形状都
返回一个 `DecisionTrace`，无法识别的部分置 `degraded=True`。

## 4. `decision_trace` 的 schema 2 用法

```
routing_schema_version  必须等于 2，否则整条轨迹标记 degraded
participation.evidence          [{code, family, strength, source}]  ← 特征来源
participation.family_contributions  {family: 贡献合计}
participation.contribution_total    宿主原始（未裁剪）加性分数 ← 阈值回放的关键
participation.score / level / should_reply
recipient.{ids, bot_targeted, confidence, threshold, ambiguous}
topic.{topic_id, confidence, threshold, margin_threshold, ambiguous}
identity.{bot_reference, mention, vocative, subject}
state.{pending_hover, active_interlocutor, intervening_users, waiting_for_answer,
       last_bot_was_question, last_bot_message_id}
mode, weights_version
```

两条依赖需要特别说明：

1. **`contribution_total` 是回放的全部依据。** 宿主把它写成「各条证据贡献之和」，
   也就是 `ParticipationPolicy.evaluate` 裁剪前的分数。本插件用它复现任意
   `strong_addressivity_threshold` 下的判定，而不重新推导任何权重。若本体改为不写
   这个字段，阈值回放会退化为不可用。
2. **`level` 不是「是否回复」。** 宿主把 `should_reply` 恒置为 `null`（没有最终发送
   决策）。本插件的 `reply` 任务因此以「`level == strong`」这一**路由准入**为预测目标，
   并在控制台与报告里都标注了这一点。

## 4b. 话题候选集（P0 数据完整性）

Topic Learner 目前看到的是「最终归属 + 候选集 + 每个候选的证据」。缺了候选集，
它就无法区分两种完全不同的错误：

```text
正确话题没进候选集        → 候选生成（embedding / 检索）的问题，改阈值没用
正确话题进了候选集但没选中 → 打分 / 排序 / 阈值的问题
```

因此 `routing.topic_candidates` 需要**稳定写入**，并建议升级为结构化形式：

```json
{
  "routing": {
    "topic_candidates": [
      {
        "topic_id": "A",
        "final_score": 0.68,
        "rank": 2,
        "evidence": {
          "semantic": 0.74,
          "reply_edge": 0.0,
          "participant_overlap": 0.81,
          "recency": 0.62,
          "lexical": 0.43
        }
      },
      { "topic_id": "B", "final_score": 0.72, "rank": 1, "evidence": { "...": 0.0 } }
    ],
    "selected_topic": "B"
  }
}
```

本插件对两种形状都接受，因此本体可以分步升级：

| 字段 | 必需性 | 缺失时的行为 |
| --- | --- | --- |
| `topic_id` | 必需 | 该条候选被丢弃 |
| `final_score` | 强烈建议 | 候选仍计入 Recall@K，但**不参与阈值回放**：把缺失分数当作 0 会凭空发明一个宿主从未有过的理由 |
| `rank` | 可选 | 按 `final_score` 降序推断（旧版 `[[score, id]]` 就靠这个） |
| `evidence` | 可选 | 只影响可解释性，不影响现有指标 |
| `routing.selected_topic` | 可选 | 回退到 `predicted_topic` |

### 两种「没有候选」不是一回事

```text
键不存在      → 本体没有记录候选集：无从判断正确话题是否被提出
空列表 []     → 本体找过，一个都没提出来：就是一次候选生成缺失
```

把两者并起来，会把每一条「无法归因」读成「检索失败」——那是数据没有支持过的结论。
插件在入库时就把这个事实单独存下来（`decision_trace.topic_candidates_recorded`），
因为只留 payload 的话，`[]` 和「没有这个字段」会长得一模一样。

早于该标记写入的旧样本没有这个字段，它们保持原来的读法（空 = 未记录），**不会**被
追溯判成候选生成失败：记录说不清楚的事，就不要替它下结论。

### 归因桶

每条话题标注恰好落入一个桶，因此归因表的合计等于话题样本总数，不会悄悄变小：

| 桶 | 含义 |
| --- | --- |
| `correct` | 选中了正确话题 |
| `candidate_miss` | 记录了候选集，但正确话题不在里面 → 候选生成 |
| `ranking_error` | 正确话题在候选集里，却被别的选中 → 打分 / 排序 |
| `not_recorded` | 没有记录候选集，无从归因 |
| `new_topic_expected` | 标注为 `NEW`，按定义不可能是候选 |
| `unattributable` | 没有可用的真值标签 |

`Candidate Recall@K` 与 `Selection Accuracy` 的分母**故意不同**：召回的分母是「真值
可归属且本体记录了候选集」的样本，选择准确率的分母**只有正确话题确实进了候选集**的
那些。这样「检索差但排序好」不会把选择准确率一起拖下去。没有样本可选时它是 `null`
（页面显示「无法计算」），而不是 0——没有可评的东西和考了零分是两件事。

无法解析的候选条目会被丢弃并**计数**（`dropped_entries`），不会静默地改变名次。
本体当前只记前 3 个候选，因此 `Recall@5` 与 `Recall@3` 必然相同——报告会把观测到的
候选长度分布打出来，免得这个截断被读成一个测出来的数字。

覆盖率的当前影响会直接写进控制台：如果没有候选集，页面会说明「无法区分候选生成与
排序问题」，而不是给出一个笼统的「话题识别错了」。

## 5. 证据词表

`core/trace.py` 冻结了 `EVIDENCE_CODES` / `EVIDENCE_FAMILIES`，镜像本体的
`core/participation_policy.py`。本体新增证据码时：

- 特征层：新码会作为未知族保留并置 `degraded`，不会静默丢弃；
- `EXPLICIT_CODES` 未收录的新**结构化**码会导致该轮被当成环境层轮次参与拟合。
  这是唯一需要人工跟进的升级点，本体新增短路证据码时应同步这里。

## 6. 本插件写什么

只写自己的共享首选项，全部以 `learning_` 开头：

| 键 | 内容 |
| --- | --- |
| `learning_index_v1` | 每会话样本数与更新时间 |
| `learning_samples_v1_<sha256(session)>` | 该会话的学习样本（无正文） |
| `learning_policies_v1` | 策略版本记录与状态 |
| `learning_state_v1` | 最近一次分析与导入的时间戳、报告摘要 |

不存在任何指向 `ysyhlly/astrbot_plugin_chat_dynamics` 作用域的写入调用。控制台里的
「采纳」只改 `learning_policies_v1` 里的状态字段。
