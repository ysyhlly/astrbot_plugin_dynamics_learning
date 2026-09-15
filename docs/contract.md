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

### 2b. 作用域身份（PR1）

`panel_runtime_v1.sessions[]` 同时带 `session_key / group_id / umo / bot_id`，本插件自
v0.6.1 起读取它们，但**只用于诊断**，不参与任何聚合：

| 字段 | 本体的行为 | 本插件怎么用 |
| --- | --- | --- |
| `session_key` | `unified_msg_origin or group_id` | **唯一的学习作用域**，`scope_hash = sha256(session_key)` |
| `umo` | `unified_msg_origin or session_key`，且本体恢复时要求 `umo == session_key` | 只记录是否等于 session_key（`scope_source`） |
| `group_id` | 平台的原始群号，**不带平台前缀**；事件没有群号时会写成 session_key | `group_hint_hash`，仅诊断，**不得作为聚合键** |
| `bot_id` | 机器人自身账号 | 暂不使用 |

三条约束，改代码时不要绕过：

1. **正常事件路径下 `umo == session_key`**，所以 `umo` 无法证明"两个会话是同一个群"。
2. **`group_id` 不是跨会话群身份**：两个适配器可能给出同一个原始群号，而本体自己的
   测试就用 `platform-a:GroupMessage:room` 与 `platform-b:GroupMessage:room` 验证过这一点；
   同一个群号还有可能来自创建 runtime 的不同代码路径。按它聚合会把两个无关会话并成
   一个画像。
3. **`scope_hash` 必须逐字节等于 `session_hash(session_key)`**。这是升级不变量：换前缀
   重新哈希会把每一份已存样本劈成两个身份。

等本体提供一个语义明确的跨会话会话标识（并说明跨 session / 跨 adapter / 跨 bot 账号
是否相同、生命周期如何）之后，改 `core/scope.py` 一处即可，样本层不需要动。

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
| `expected_reply` | bool | 回复**两个**任务的共同标签：准入任务与最终发送任务 |
| `routing.topic_confidence` / `routing.topic_ambiguous` | float/bool | 话题阈值回放 |
| `routing.topic_candidates` | list | **放宽**归属阈值的回放；缺失时该方向不可回放 |
| `decision_trace` | dict | 证据族、贡献值、身份与状态特征 |

字段缺失一律降级，不抛异常：`core/trace.py` 的 `parse_decision_trace` 对任何形状都
返回一个 `DecisionTrace`，无法识别的部分置 `degraded=True`。

## 4. `decision_trace` 的 schema 用法（2 与 3）

读取端实现两个 reader 加一层归一化，而不是「有什么读什么」：

| reader | 触发条件 | 产出 |
| --- | --- | --- |
| `read_schema_v2` | `trace_schema_version == 2` | 证据、分数、话题判定；**没有**最终结果 |
| `read_schema_v3` | `trace_schema_version == 3` | 上述全部 + 候选逐条证据 + 选择结果 + 最终发送结果 |
| `normalize_trace` | 按记录的版本分发 | 版本缺失或不认识时按 schema 2 字段集读取，并置 `degraded` |

两条规则：

1. **自描述字段照读。** `routing` 与 `outcome` 的键名本身就说明含义，即使版本号没跟上也会被读取并计入 —— 丢掉本体真写过的事实比版本不符更糟。版本缺失/不认识时轨迹标 `degraded`，报告里会说出来。
2. **版本决定证据等级。** 只有声明了 schema 3 的记录才可能达到 `candidate_evidence = full`；键名再像也不能把旧记录提升到它从未声明的完整度。

写回时写的是**读到的那个版本**，所以 schema 2 的样本往返之后仍然是 schema 2，不会凭空长出本体没写过的字段。

### 两个显式降级标记（P0 契约）

schema 2 记录会被明确标成两件事，而不是静默留空：

| 标记 | 含义 |
| --- | --- |
| `outcome_unavailable` | 没有任何字段能回答「最后到底发出去了没有」 |
| `candidate_evidence_partial` | 候选集可能有，但逐条候选证据不完整 |

`candidate_evidence` 是三值：`full` / `partial` / `none`。`none` 比 `partial` **更**降级（连候选集都没有），两者都让 `candidate_evidence_partial` 为真 —— 需要区分的读者看三值字段。

### 轨迹 schema 的键名

轨迹里的 schema 号在 **v1.6.2 起写作 `trace_schema_version`**。旧键名 `routing_schema_version`
仍然读取 —— 它描述的是整条轨迹（收件人、话题、参与度，以及 schema 3 起的最终结果），
不只是 routing 段，名字说错了事；但已经落库的标注不能因此读成「未记录」。

两个键都存在时以 `trace_schema_version` 为准；本插件写回时只写新键名。

### schema 2 字段表

```
trace_schema_version    2 或 3，其它值整条轨迹标记 degraded
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
2. **`level` 不是「是否回复」。** 宿主初始 `should_reply` 为 `null`；规则模式在门禁后
   可写入准入布尔值，它仍不证明最终发送成功。因此「回复」被拆成两个任务：**回复准入**以「`level == strong`」为
   预测目标，**最终发送**以 schema 3 的 `outcome.delivered` 为预测目标。只有前者是路由
   判定；被作息压掉的发送属于后者，不会被算成路由漏回复。

### schema 3 新增字段

```json
{
  "trace_schema_version": 3,
  "routing": {
    "selected_topic": "B",
    "topic_candidates": [
      {
        "topic_id": "A",
        "final_score": 0.68,
        "rank": 2,
        "evidence": {
          "centroid": 0.74,
          "exemplar": 0.70,
          "recent": 0.68,
          "lineage": 0.0,
          "participant": 0.81,
          "recency": 0.62,
          "lexical": 0.43
        }
      }
    ]
  },
  "outcome": {
    "final_outcome": "delivered | suppressed | generation_failed | delivery_failed | not_attempted",
    "delivered": true,
    "suppression_reason": "asleep_ambient",
    "stage": "gate | generation | delivery | admission"
  }
}
```

| 字段 | 必需性 | 缺失时的行为 |
| --- | --- | --- |
| `routing.selected_topic` | 建议 | 回退到 `predicted_topic` |
| `routing.topic_candidates[].evidence` | 建议 | 候选仍计入召回，只是拿不到 `candidate_evidence = full` |
| `outcome.final_outcome` | 建议 | 回退到 `delivered` / `suppression_reason` 推导；都没有就是「未记录」 |
| `outcome.stage` | 可选 | 按 `suppression_reason` 查表；查不到就是 `unknown`，**不会**算成门禁压制 |

`outcome` 写在 `decision_trace` 里或写在标注记录同级（`record["outcome"]`）都可以：
最终结果是在快照冻结之后才知道的，写到哪一层都接受，读取顺序是 trace 优先、记录兜底。

**抑制原因是开放词表。** 本体新增 `reason_code` 时，优先保留宿主明确声明的有效阶段；
没有声明阶段且本地词表也不认识原因时，记成 `unknown`。已知词表镜像
`decision_gate` / `arbiter` 实际产出的字符串，见 `core/outcome.py`。

**没有结果 ≠ 没有发送。** `delivered: false` 且没写原因时，值是 `not_delivered`、阶段是
`unknown`：在「被压掉」和「生成失败」之间替本体做选择，就是本插件在发明一个本体没说过
的原因。

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
          "centroid": 0.74,
          "exemplar": 0.70,
          "recent": 0.68,
          "lineage": 0.0,
          "participant": 0.81,
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
| `learning_state_v1` | 最近一次分析与导入的时间戳、报告摘要、诊断，以及契约面计数快照（`last_contract_stats`） |

不存在任何指向 `ysyhlly/astrbot_plugin_chat_dynamics` 作用域的写入调用。控制台里的
「采纳」只改 `learning_policies_v1` 里的状态字段。

### 发布面：`policy_contract_version`

策略状态机是 `proposed → validated → shadow → promoted → superseded / rolled_back`。
只有 `promoted` 的记录会出现在 `GET /published`：

```json
{
  "policy_contract_version": 1,
  "generated_at": 1760000000.0,
  "policies": [
    {
      "policy_contract_version": 1,
      "policy_id": "policy_v3",
      "issued_at": 1760000000.0,
      "state": "promoted",

      "source": {
        "trace_schema_version": 3,
        "trace_schema_versions": { "2": 20, "3": 100 },
        "dataset_fingerprint": "9f2c…",
        "learning_version": "0.9.0"
      },

      "target": {
        "chat_dynamics_version": "1.6.2",
        "baseline_config_hash": "3ab41f0c9d2e7b85",
        "validated_host_versions": ["1.6.2"]
      },

      "params": { "strong_addressivity_threshold": 0.67, "...": "全部参数，不只是变动的那个" },
      "baseline": { "...": "..." },
      "changed": ["strong_addressivity_threshold"],
      "target_error": "missed_bot",
      "confidence": "moderate",
      "shadow_observed": false,
      "evidence": { "holdout": {}, "forward": {}, "collateral_regressions": [] }
    }
  ],
  "note": "本插件只发布策略，不写入 ChatDynamics 任何配置；是否采用由本体决定。"
}
```

发的是**解析后的全部参数**而不是增量：只发增量的消费者得自己补全其余项，而那个补全
会静默变成它实际应用的值。`shadow_observed` 说明这份策略有没有经过影子观察 —— 没有
不是失败，它是「赢在离线」和「被看着跑过」的区别，本体有权知道拿到的是哪一种。
当前策略记录没有持久化与该策略匹配的真实影子流量证据，因此 `shadow_observed`
保守返回 `false`；前向留出集和状态切换均不能证明在线观察。`shadow_entered` 仅表示
状态历史曾进入影子阶段。实际流量和带标签比较请查看 `/shadow`。

### 三个版本号，互不推导

```text
policy_contract_version      这个文件本身的协议版本        Learning -> ChatDynamics
source.trace_schema_version  策略是在哪种 trace 上训出来的  ChatDynamics -> Learning
source.learning_version      哪个 Learning 版本产出的
target.chat_dynamics_version 它是对着哪个本体版本验证的
```

三条不允许的推论：

1. **读取端升级不能动 `trace_schema_version`。** 那个数字描述的是**本体写什么**，
   只有本体能改。本插件改 reader、API 或页面都不影响它；
2. **不能从 SemVer 推断兼容。** `1.7.0 → 1.7.1` 可能改掉一个参与度计算、topic score
   或门禁顺序，而这份文件里每一个阈值都是对着旧分布校准的。所以
   `validated_host_versions` 是**验证出来的列表**（今天只含训练时观测到的那一个版本，
   本体没报版本时为空），不是 `major_minor_equal()` 这种猜测；
3. **本体没报版本时它就是空的。** Learning 无从知道眼前的 trace 是哪个本体版本写的，
   空列表是「无法验证」，不是「匹配」。

本体的消费侧规则（**这三条由 ChatDynamics 实现**）：

| 模式 | 版本不匹配时 |
| --- | --- |
| `active` | 拒绝应用。要求 host 版本 ∈ `validated_host_versions`（今天是严格相等） |
| `shadow` | 允许读取用于展示 / 评估，但必须打上 `version_mismatch`，**绝不实际应用** |

### `dataset_fingerprint` 是来源证明，不是运行时兼容条件

`source.dataset_fingerprint` 回答的是「这份策略由哪批训练 / 验证数据产生」。
它是 audit metadata：

- **必须存在**于策略记录里，并随 `/published` 发布、在控制台展示；
- **不因为缺失或不同而拒绝**。要求用户在本体侧再手填一次同一个指纹，等于
  「Learning 发布 ABC、用户去 ChatDynamics 手填 ABC、本体比较 ABC == ABC」——
  安全收益有限，操作成本不低；
- 只有消费端**显式配置** `expected_dataset_fingerprint` 时才做 pin 校验，且必须完全相等。

真正适合「人为批准锁定」的是 **`policy_id`**：

```json
{ "learning_policy_expected_policy_id": "policy_v3" }
```

管理员表达的是「我批准运行 policy X」，而不是「我批准训练 policy X 的那批数据 hash」。

### `baseline_config_hash`：跨仓库的规范化约定

`target.baseline_config_hash` 是本体侧用来发现「配置动了」的摘要。规范形式属于契约，
不是实现细节 —— 两边必须一致：

```text
sha256('{"parent_accept_threshold":"0.7200","safe_hover_threshold":"0.4000",...}')[:16]

    参数名全部列出，按字典序排序
    每个值固定四位小数
    分隔符 "," 与 ":"，不含空白
```

小数位或键序不同，会让同一个配置产出不同的摘要 —— 那会被读成「配置变了」，而它并没有。
截断到 16 个十六进制字符：它用来发现漂移，不用来鉴权。

## 7. 契约健康度：为什么必须在原始记录上数

`GET /quality` 把「本体到底写了什么」和「学习层能用这些样本做什么」分开报告。

**这里曾经有个命名事故，已经修掉。** 旧字段叫 `contract_version`，同时表示两件不相干的
事：本体写的 trace schema，以及本插件读取端的修订号。结果是读一次就要猜这个 4 指的是哪
一个。现在拆成三个互不推导的名字：

| 名字 | 含义 | 谁改它 |
| --- | --- | --- |
| `trace.supported` / `trace.latest` | 本插件**能读**哪些 trace schema | 只有本插件能读的集合变化 |
| `trace.observed` | 本体**实际写了**哪些（分布） | 只有本体 |
| `reader_version` | 本插件自己的读取端修订号（v0.9.0 起重置为 1） | 本插件 |

`trace_schema_version` 这一个概念在本插件内部只有一个名字
（`DecisionTrace.trace_schema_version`，来源是本体的 `decision_trace.trace_schema_version`）。
本体没有升级时它就是 2，本体升到 schema 3 时它就是 3 —— 与本插件发什么版本无关。

分开的原因不是分层好看，而是**样本层是有损的**：

| 原始记录里的事实 | 经过 `core/trace.py` 归一化之后 |
| --- | --- |
| `decision_trace.trace_schema_version = 1` | 按 schema 2 字段集读取并写回 `2`，同时置 `degraded`、`source_schema=1` |
| `decision_trace.trace_schema_version = 3` | 按 schema 3 字段集读取并**写回 3**（v0.9.0 起不再一律压成 2） |
| `participation.contribution_total = null` | 永远读回 `0.0`（`_finite` 把缺失归一成零） |
| `routing.topic_candidates` 键不存在 | 由 `topic_candidates_recorded` 标记单独带走（v0.6.0 起） |
| `contribution_total` 键不存在 | 由 `contribution_total_recorded` 标记单独带走（v0.6.1 起） |
| `outcome` 键不存在 | 由 `outcome_unavailable` 标记单独带走（v0.9.0 起） |
| 候选条目没有 `evidence` | 由 `candidate_evidence`（`full`/`partial`/`none`）单独带走（v0.9.0 起） |

所以对 `/quality` 的规定是：

1. **契约面**只在导入时对原始记录计数，并把快照连同 `contract_at` 一起返回——它永远是
   「上一次导入时的样子」，页面必须按这个口径读；
2. **样本面**每次请求实时重算，判据取自学习器自己的谓词（`core/policy.py` 的回放分支、
   `core/topic_learner.py` 的 `replay_can_move`），不在这里二次推导；
3. 契约面的会计恒等式为

   ```text
   annotations_seen == annotations_kept + malformed + unknown_session
   ```

   并由 `balanced` 字段自报。不守恒时它先说自己是坏的，而不是让读者去猜；
4. 字段类计数（schema 分布、候选桶、加性分数有无、最终结果有无）只统计**进入了样本的
   那部分记录**，被丢掉的行由会计类计数负责。分子的分母不能来自另一个人群。
## 8. 回复的两层：准入与最终发送（P0）

Schema 2 只能回答「路由准不准」。把「该回但被作息压掉」当成一次 `missed_reply`，
会让报告去动一个没有错的阈值。所以回复被拆成两个任务，各有各的预测目标：

| 任务 | 预测目标 | 回答的问题 | 分母 |
| --- | --- | --- | --- |
| `reply_admission` | `participation.level == strong` | 该不该进入回复流程 | 所有带 `expected_reply` 的标注 |
| `reply_outcome` | `outcome.delivered` | 最后到底有没有发出去 | 只统计**记录了结果**的标注 |

三条不可绕过的规则：

1. **没有结果就不产生 `reply_outcome` 样本。** schema 2 数据下这个任务为空，而不是
   一整列编出来的 `silent`。缺失的标签永远不是负样本。
2. **最终发送层不可回放。** 门禁、生成与平台发送都不在记录轨迹里，所以这一层只报告
   已记录的事实，不参与任何阈值回放，也不参与「可采纳」的门禁结论。
3. **同名错误不许复用。** 准入层的漏回复叫 `missed_reply`，最终发送层的漏发送叫
   `undelivered_reply`：同一个计数器不能既表示「路由该回没回」又表示「该发没发」。

`reply_outcome` 样本的 `error_type` 直接写清停在哪一环：`gate_suppression` /
`generation_failure` / `delivery_failure` / `participation_error`；阶段认不出来时写
`unattributable`，不往最近的桶里塞。

## 9. 错误归因链（P0）

每条**被标注的消息**（不是每条样本）归入唯一一格，合计等于语料规模：

```text
recipient_error        机器人没被认成收件人
topic_candidate_miss   正确话题没进候选集（生成 / 检索）
topic_ranking_error    正确话题进了候选集但没被选中（打分 / 排序）
participation_error    该不该进入回复流程判断错了（参与准入）
gate_suppression       准入没错，被作息 / 降温 / 媒体压掉（不是路由错误）
generation_failure     进了流程，生成没产出可用回复（不是路由错误）
delivery_failure       生成了，平台发送失败（不是路由错误）
unattributable         记录不足以判断哪一环出错
ok                     这条链上没有发现错误
```

三条设计约束：

1. **这是复查顺序，不是因果结论。** 「谁在跟谁说」「在聊什么」「要不要插话」「有没有
   发出去」是四个基本独立的层，一层错不会导致下一层错。顺序只回答「先看哪一层」。
2. **一条消息只记一格。** 同时错在两层的消息，第二层记进 `also_failed`，不重复计数 ——
   重复计数会让桶加起来超过语料规模，那正是归因表开始说谎的方式。
3. **门禁压制只在原因写明时才算门禁。** 未知的 `suppression_reason`、或只有
   `delivered: false` 而没有原因，都归 `unattributable`：不知道停在哪一环，就说不知道。

话题那一格复用 `core/topic_learner.candidate_observations` 自己的判定，不在这里二次
推导 —— 两份实现会在同一条消息上给出两个答案，而两个都印着数字。

接口：`GET /astrbot_plugin_dynamics_learning/attribution`（`?examples=N` 控制样例条数）。

## 10. Shadow A/B：分歧子集与进入 active 的门槛（v1.0.0）

本体的 `shadow` 段（见 ChatDynamics 侧 `docs/learning-contract.md`）带来一次比较：
同一轮消息，baseline 判了一次，策略判了一次。

    {
      "policy_id": "policy_v3",
      "baseline_threshold": 0.70, "shadow_threshold": 0.67,
      "baseline_reply": false, "shadow_reply": true,
      "changed": true,
      "score": 0.68, "baseline_margin": -0.02, "shadow_margin": 0.01,
      "reason": "ambient" | "structural" | "early_return",
      "recorded_at": 1760000000.0
    }

### 为什么要分组、为什么要成对

一条消息最多产出四个样本，按样本读会把同一次比较最多算四遍，分歧率会随人工标了几个标签
而变。所以 `shadow_rows` 按 `(session, msg_id)` 分组，标签只从准入样本上取。

配对表是**四格划分**，不是只数分歧：

```text
                 shadow 对        shadow 错
baseline 对      both_correct     baseline_only    ← 策略的代价
baseline 错      shadow_only      both_wrong
                 ↑ 策略的收益
```

两个非对角格正好是分歧子集，`changed == baseline_only + shadow_only` 恒成立；
`balanced` 字段自报这个恒等式。

### 门槛（`learning_shadow_*` 可调）

| 检查 | 默认 |
| --- | --- |
| `shadow_samples` | ≥ 500 条带标签 |
| `disagreements` | ≥ 100 条分歧 |
| `overall_regression` | ≥ −1% |
| `target_improvement` | 绝对 ≥ 1%，或相对错误率改善 ≥ 10%（且基线错误率 ≥ 5%） |
| `interval_floor` | 95% 区间下界 ≥ −0.2% |
| `sessions` / `active_hours` | ≥ 3 个会话 / ≥ 4 个活跃时段 |
| `subgroups` | 支撑 ≥ 20 的会话回退不得 > 5% |
| `net_gain` | > 0 |

每项都自报测量值：只报失败的门槛读不出「没有失败」和「什么都没跑」的区别。
活跃时段用 `shadow.recorded_at`（本体做判定的时间），不是样本时间戳（人工标注的时刻）。

### 覆盖率的边界

本体对**每条**处理过的消息都记录了比较，但只有人工标注过的会成为样本。契约面因此多了
`shadow.present` / `shadow.absent` 两个计数 —— 不这样，表里的 320 会被读成总体的规模。
要一个真正的总体分歧率，需要本体把每条决策单独记一份日志，那是另一个契约增量。

## 11. 采纳门槛与数据门槛（P1）

**采纳需要两个留出集都通过**，加上一个不跨 0 的区间：

```text
会话留出集   它不是在背某一段对话        （按 session 切）
前向留出集   它没有在群改变习惯后失效     （按标注时间切：旧段训练、新段验证）
置信区间     提升比重采样噪声大          （按会话成对 bootstrap，默认 95%）
```

任一条跑不起来时结论是 `insufficient`，不是 `accepted`：「没看」和「看了没问题」
不能给出同一个结论，否则门槛就是装饰。自动路径最多到 `validated`；`promoted` 需要
人点一下，或者本体真的做过影子观察。

**数据门槛**在任何学习之前跑，不通过就完全不产生策略记录（连 `proposed` 都不记，
否则控制台里会躺着一个等着被点的版本号）：

| 检查 | 阻塞？ | 为什么 |
| --- | --- | --- |
| `samples` / `sessions` | 是 | 样本太少、会话太少时统计本身不成立 |
| `label_age` | 是 | 标注太旧：群的习惯可能已经变了 |
| `balance` | 否 | 正类比例过低会让 F1 失去分子，但只限定结论强度 |
| `degraded` | 否 | 轨迹降级说明契约没对齐，先修契约 |
| `candidate_coverage` / `outcome_coverage` | 否 | 只限定对应方向：候选覆盖低影响话题，结果覆盖低影响回复 |

分组（当前 = 会话）分层评估只做诊断，**不产生本地策略**：在一个会话上拟合出的阈值
是记住对话，不是学会相处。


### Shadow 候选读取面

`GET /astrbot_plugin_dynamics_learning/candidate` 和本插件 KV
`learning_candidate_v1` 提供 `validated`、`shadow`、`promoted` 策略，供本体
shadow consumer 观察；不需要提前 promote。顶层 `eligible_modes=["shadow"]`。
`GET /published` 与 `learning_published_v1` 继续仅提供 `promoted`，供 active
consumer 使用。每个策略携带 `eligible_modes`：非 promoted 仅 shadow，promoted
允许 shadow/active；本体仍须独立校验状态、身份、pin 和兼容性，并保证 shadow 永不应用。

两份 KV 在插件初始化、分析结果写入和状态变更时从当前策略记录重新生成；回滚与
替代会撤回候选，清空存储同时删除两份 KV。该读取面不会写入本体配置。

### 读取兼容与缺失字段

缺少 `accepted_from` 时标注来源未知，不计为纯人工；仅明确 `human` / `manual` 计入
`human_only`，明确 `ai` 计入 `ai_assisted`。不能由宿主版本号推断它是否写过该字段。

运行快照版本与 trace schema 独立。读取器对 `sessions[].session_key` 做结构校验，未知运行版本仍可恢复会话，同时通过 `runtime_version` / `runtime_version_supported` 报告版本差异。

当前宿主运行快照未写出 `graph` 限额；缺失时窗口返回 null 限额、截止时间和容量压力，不把默认值当作实际配置。

最终结果优先取 trace，其次取标注记录，`source` 分别为 `trace` / `record`。宿主显式报告的有效 `stage` 优先于本地 reason 字典；未知 reason 且没有显式阶段时仍为 unknown。shadow trace 保留判定时的 `recorded_at`，缺失时不补造时间。
