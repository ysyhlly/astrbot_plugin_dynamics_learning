# 群间 · Dynamics Learning

> ChatDynamics 的**行为学习层**。让群聊系统根据真实群聊里的人工标注和长期运行结果，
> 逐渐学会更准确地判断话题、对话对象和参与时机。

这是 [astrbot_plugin_chat_dynamics](https://github.com/ysyhlly/astrbot_plugin_chat_dynamics)
的附属插件，对本体**只读**。

```text
LivingMemory     = 记住世界
Self Learning    = 学会怎么表达
Dynamics Learning= 学会怎么相处      ← 本插件
ChatDynamics     = 决定此刻该怎么参与
```

---

## 它做什么，不做什么

做到的：

- 把 ChatDynamics 已有的**人工标注**与**决策轨迹快照**转成统一、可回放的 `LearningSample`；
- 分别统计**对话对象**、**话题**、**回复准入**三个任务的误差模式与证据倾向；
- 用**离线回放评测**对比 baseline 与 candidate，只有候选在留出集上胜出才标记为可采纳；
- 把参数建议绑定到 ChatDynamics 的**真实配置键**，并按硬上限截断。

**不做的**（这是设计约束，不是未完成项）：

- 不自动修改 ChatDynamics 的任何配置；
- 不向 ChatDynamics 的共享首选项写入任何内容；
- 不保存消息正文；
- 不训练神经网络模型——第一版只需要规则统计、逻辑回归、贝叶斯先验和阈值优化。

控制台里的「采纳」只写入本插件自己的策略记录。

---

## 安装

1. 确认 `astrbot_plugin_chat_dynamics` 已安装并运行过（本插件读它的共享首选项）；
2. 把本目录放进 AstrBot 的 `data/plugins/`，重载插件；
3. 打开插件页 **Dynamics Learning**。

无第三方运行时依赖：学习与评测全部是标准库实现，拟合是确定性的（相同输入必然得到
相同权重），这是策略可复现的前提。

### 配置要点

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `source_plugin_id` | `ysyhlly/astrbot_plugin_chat_dynamics` | 本体在共享首选项里的作用域标识 |
| `learning_max_samples` | 5000 | 按会话分片，超出后淘汰最久未更新的会话 |
| `learning_min_samples` | 100 | 低于此数只出统计与诊断，不出参数建议 |
| `learning_max_param_delta_ratio` | 0.05 | **单次参数变化硬上限**（±5%），调大也不会超过 20% |
| `learning_min_evaluation_samples` | 40 | 留出集样本不足则结论为「样本不足」 |
| `learning_min_improvement` | 0.02 | 判定可采纳所需的提升 |
| `learning_max_regression` | 0.01 | 任一核心指标回退超过此值即拒绝 |
| `learning_holdout_ratio` | 0.30 | 按**会话**切分的留出比例 |
| `learning_store_raw_trace` | true | 保存无正文的决策轨迹快照，便于日后重新提取特征 |

---

## 快速开始

```text
1. 在 ChatDynamics 的「场景回放」页对若干消息做人工标注
   （expected_topic / bot_targeted / expected_reply / 收件人纠错）
        ↓
2. 本插件控制台点「导入标注」
   → 标注被转成 LearningSample 并分片存储
        ↓
3. 点「运行分析」
   → 统计 + 证据倾向 + 参数建议 + 离线评测
        ↓
4. 看「系统建议」与「离线评测」
   → 只有评测判定 accepted 的建议才是可采纳的
        ↓
5. 在「策略版本」里采纳 → 记录一条策略版本
   → ChatDynamics 配置不会发生任何变化
```

没有真实本体时也可以用「导出/导入」：控制台「导出 JSON」拿到的结构可以直接通过
`POST /astrbot_plugin_dynamics_learning/ingest` 且 `source=export` 回灌，便于离线
复算与回归。

---

## 三个任务

### recipient —— 这句话在对谁说

优先级最高。样本来自 `bot_targeted` 标注与轨迹里的收件人证据。

- **统计**：人工样本上的准确率、混淆矩阵、错误类型分布（`missed_bot` / `false_bot` /
  `wrong_recipient` …）；
- **证据倾向**：每条环境层证据码的贝叶斯平滑出现率与相对基准的 lift；
- **可采纳建议**：在**记录分数**上扫描 `strong_addressivity_threshold`，按 ±5% 截断；
- **诊断（不可导出）**：拟合的环境层逻辑回归评分——它在本体开放「环境层评分替换」之前
  无法变成配置项，所以只报数字、不进候选。

结构化轮次（显式 @、引用、点名叫、平台唤醒）由宿主短路决定，与阈值无关，因此既不参与
拟合也不受候选影响。无前序 Bot 消息的提前返回同理。

### topic —— 这条消息属于哪个话题

**指标**：会话内配对计算的 `wrong_merge`（误合并）与 `fragmentation`（误拆分），
对标签重命名不变；预测为「未归属」的样本按本体口径计入误拆分。

**候选归因**：一个错误的话题决策其实是两种，改法完全相反：

```text
Candidate Recall@3   96.2%   Selection Accuracy  83.7%  → 优化打分 / 阈值
Candidate Recall@3   78.1%   Selection Accuracy  91.4%  → 优化候选生成（embedding / 检索）
```

- `Candidate Recall@K`：正确话题是否出现在候选集中；
- `Selection Accuracy`：**只在正确话题已进入候选集的样本上**，宿主选中的比例。

两者条件不同，所以不会互相掩盖。逐条归因分为 `candidate_miss`（候选生成缺失）、
`ranking_error`（候选在但没选中）、`not_recorded`（快照没存候选集，不可归因）。

**可采纳建议**：只在**误合并占主导**时收紧 `topic_commit_threshold`，因为收紧只会
移除归属，可以完整回放。

**诚实边界**：放宽归属需要知道被拒绝的候选话题分布，只有标注快照里存了
`routing.topic_candidates` 的消息才能重构。当误拆分占主导而候选覆盖不足时，本插件
给出**方向**并指名本体需要补的字段，而不是给一个无法验证的数字。本体侧的字段规格见
[`docs/contract.md`](docs/contract.md) 第 4b 节。

### reply —— 此刻要不要接话

宿主没有记录最终发送决策（`should_reply` 恒为 `null`），所以这里的预测目标是
**路由准入判定**（`participation.level == "strong"`），不是「真的发出去了」。
指标为 precision / recall / F1，F1 同时是评测里的回退守卫。

---

## 迭代调参

一次 ±5% 的参数变化很难单独带来 2% 提升，真实群聊噪声只会让这件事更难。所以预算花在
**一次迭代能走多远**，而不是一次能跳多远：

```text
单步变化上限    ±5%    相对于当前值
累计漂移上限    ±15%   相对于原始基线
最大连续步数    3      （0.95^3 = -14.3%，与累计上限自然吻合）
```

```text
Baseline
   │
   ▼  精确 Replay（留出集）
 Safe? ── 否 ──> Reject / 回滚
   │ 是
   ▼
 Promote? ── 是 ──> 进入 learned policy
   │ 否
   ▼
 边际 ≥ +0.5%？── 是 ──> 再走一小步（回到 Replay）
   │ 否
   ▼
 连续两步 < +0.2% ──> 停止，保留已获得的收益
```

| 级别 | 条件 | 含义 |
| --- | --- | --- |
| **Safe candidate** | 累计 ≥ -0.2%，目标错误改善，无核心指标回退 | 采纳本步，可以再走一步 |
| **Advance** | 边际 ≥ +0.5%，或目标错误相对下降 ≥5% | 值得再走一步 |
| **Promote** | 累计 ≥ +1.0%，或目标错误相对下降 ≥10%（且累计 ≥ -0.2%） | 可进入 learned policy |
| **Strong promote** | 累计 ≥ +2.0% | 高置信度 |

止损：

- 任一步核心指标回退 > 1% → 停止并**整轮回滚**到基线；
- 连续两步边际收益 < +0.2% → 停止，保留已获得的收益；
- 累计漂移触及 ±15% 且仍在改善 → 停止并**要求人工确认**。

**2% 不是被删掉，而是换了角色**：从"每次采纳的硬门槛"变成"非常确定值得升级"的强信号。

一个符合直觉的例子：

```text
step 1   0.700 → 0.665   (-5%)   留出集 +0.8%   → 继续
step 2   0.665 → 0.632   (-5%)   留出集 +0.8%   → 累计 +1.6% → Promote
```

每一步都能单独 Replay，也都单独受 ±5% 约束。**没有任何一步是被允许跳过去的。**

### 按错误类型评判

判断一次调整是否成功，看的不是总准确率，而是**它针对的那个错误**：

```text
误拆分   40 → 25   （相对 -37.5%）
误合并    5 → 12   （相对 +140%）      ← collateral，会被报告
总准确率  83.1% → 83.7%（+0.6%）       ← 只动了 0.6%，但这是一次成功的调整
```

所以每次调整都会**指明它针对的错误类型**：

| 参数方向 | 目标错误 |
| --- | --- |
| `strong_addressivity_threshold` ↓ | `missed_bot`（漏识别） |
| `strong_addressivity_threshold` ↑ | `false_bot`（误触发） |
| `topic_commit_threshold` ↑ | `wrong_merge`（误合并） |
| `topic_commit_threshold` ↓ | `fragmentation`（误拆分） |

其他错误作为 collateral 一并报告，并参与守卫。

## 离线评测怎么算

```text
学习样本
   ↓ 按会话切分（同一段对话不会同时出现在训练和验证里）
训练集 ────────────────► 留出集
   │                        │
   ├ 拟合环境层评分          │
   ├ 扫描阈值（训练集上）     │
   ↓                        ↓
candidate ─────────────► 只在留出集上打分一次
```

判定规则（与计划书一致）：

- 留出集任一任务样本 < `learning_min_evaluation_samples` → **样本不足**，不下结论；
- 任一**核心指标**回退 > `learning_max_regression` → **拒绝**；
- 主指标（recipient 准确率 / topic 配对准确率 / reply F1）提升 ≥ `learning_min_improvement`
  → **可采纳**；
- 否则 → **拒绝**。

precision 与 recall 会被报告但不参与守卫：它们天然此消彼长，同时守卫会否决任何平衡的
调整。F1 已经在守卫里，precision 崩塌一样会被拦下。

### 回放的是什么

回放的是**记录轨迹上的决策函数**，不是 ChatDynamics 路由器的完整重跑：不重新做
embedding、不重新检索父消息、不调用模型。因此留出集数字是「在这个标注集上，换一组参数
会不会判得更准」，不是生产准确率。报告里每个结果都带着这句话。

---

## 数据契约健康度（能力矩阵）

页面最上面那张表不回答「数据好不好」，它回答**这批记录能不能支撑某项分析**。
每一行是一项真实存在的能力（定向阈值回放、话题候选归因、话题阈值回放、回复准入回放、
作用域身份、最终发送结果），带自己的可用条数、合计条数、覆盖率，以及每一条被排除的原因。

常驻「不支持」的那一行是有意放在那里的：本体把 `participation.should_reply` 恒置为
`null`，所以没有任何样本记录过"最终到底发没发"。它显示为「不支持」而不是错误——
这是数据回答不了的问题，不是系统出故障；本体哪天开始记录这个字段，它就会自动变成
一项可用的能力。

| 状态 | 含义 |
| --- | --- |
| 正常 | 覆盖率 ≥ 90%，这项分析可以用 |
| 警告 | 有可用样本但覆盖不足，指标只在那部分子集上成立 |
| 不支持 | 这批数据里**没有任何**样本能行使这项能力（缺字段 / 缺标签），不是系统出错 |
| 样本不足 | 条数低于评测门槛，还不到能判断的时候 |

健康度分两个面，因为它们的证据来源不同：

- **契约面**（最近一次导入时对**原始记录**计数）：本体写了哪个 schema、有没有
  `decision_trace`、`routing.topic_candidates` 字段是否存在、`contribution_total` 是
  `null` 还是 `0`。这些事实一旦进入样本层就会被归一化抹平（轨迹永远重发成 schema 2，
  缺失的分数永远读回 `0.0`），所以只能在原始记录上数；
- **样本面**（每次请求实时重算）：学习者实际读到的那批样本能做什么，判据直接取自
  `core/policy.py` 的回放分支与 `core/topic_learner.py` 的回放谓词，不在这里二次推导。

契约面的计数满足 `seen == kept + malformed + unknown_session`，并由 `balanced` 字段
自报；不守恒时它会先说自己坏了，而不是让读者去猜。

## 会话画像

第二个标签页，按**作用域**（当前等于会话）列出「被人工检查过的样本里，系统经常错在哪里」。

两条口径写在每一段输出里，因为它们决定数字能被读成什么：

1. **只统计被检查过的样本。** 标注是人挑出来标的，所以这是选中样本的分布，不是这个
   会话的真实错误率。一个被专门翻查问题的会话天然比没人看的会话"更差"。
2. **对照的是 leave-one-out 基线。** 偏差算的是「本会话」对「除本会话以外的全部样本」。
   用包含自己的全局做分母会把差异朝 0 拉，会话越大越接近和自己比。

小样本用 Beta-Binomial 先验向基线收缩（`prior_strength = 20`）：三条样本里的两个错误
不会变成一个 66.7% 的"群性格"。原始值、平滑值、基线值三个数一起给，偏差用平滑值算，
所以任何一次变化都能解释。

置信度分四档，且「稳定」不是靠一个晚上标 100 条换来的：

| 档位 | 条件 |
| --- | --- |
| 样本不足 | < 20 条被检查样本 |
| 置信度低 | 20–99 条 |
| 置信度中 | ≥ 100 条，但标注跨度不足 |
| 稳定 | ≥ 100 条、≥ 3 个标注日，且 ≥ 3 个会话（见下） |

「稳定」的会话项跟着契约走：当前一个作用域就是一个会话，所以要求 3 个会话等于永远不达标——
那种门禁是装饰，不是保护。因此这一项按 `core/scope.py` 的 `SCOPE_SPANS_SESSIONS` 缩放，
本体哪天提供真正的跨会话群身份，它会自动收紧，画像代码不需要改。

时间维度用的是**标注时间**（`annotated_at`），不是消息发生时间——本体没有冻结消息时间，
所以它只能说明"这次复查是分散的"，不能说"这个会话长期如何"。

候选链诊断先过门禁再下结论：候选覆盖率 ≥ 70%、可算召回 ≥ 20 条、可算选中准确率 ≥ 20 条，
三者之一不满足就输出「候选证据不足」，而不是在十个样本上猜是 embedding 还是阈值的问题。
过关之后才会给出：

```text
Recall@3 低 + 选中准确率高  → candidate_generation（改检索，暂不建议动 topic_commit_threshold）
Recall@3 高 + 选中准确率低  → ranking_or_scoring（阈值与间隔才是可动的旋钮）
```

## 控制台

页面分两个视图：**总览**（学习与调参）和**会话画像**（逐会话复查）。

| 区域 | 内容 |
| --- | --- |
| 数据契约状态 | 能力矩阵、契约面计数与最近一次导入的字段缺口 |
| 学习概览 | 样本数、会话数、策略记录数、数据来源与上次导入诊断 |
| 最近 7 天标注表现 | 三个任务在窗口内的标注准确率（始终附样本数） |
| 错误分布 | 按任务与错误类型汇总 |
| 系统建议 | 参数建议（含 before→after、理由、置信度、评测结论）与工程诊断 |
| 离线评测 | baseline vs candidate 逐指标对比、结论与候选参数 |
| 环境层学习评分 | 拟合评分的留出集表现与它需要的前提（诊断，不可导出） |
| 策略版本 | 版本列表、采纳 / 忽略 / 回滚 |
| 样本浏览 | 分页查看样本（无正文，身份字段脱敏） |
| 会话画像 | 会话列表（被检查样本量、置信度、主要问题、候选链诊断）与单个会话的偏差明细 |

## Web API

全部挂在 `/astrbot_plugin_dynamics_learning/` 下，走宿主 Dashboard 的插件鉴权：

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `overview` | GET | 状态、配置、数据面、上次导入诊断 |
| `samples` | GET | 分页样本（`page` / `page_size` / `task` / `session` / `scope`） |
| `quality` | GET | 数据契约健康度：能力矩阵 + 契约面计数（含 `contract_at`） |
| `scopes` | GET | 作用域列表：被检查样本量、置信度、主要问题 |
| `scope` | GET | 单个作用域画像：`profile` / `global`（LOO 基线）/ `deltas` / `diagnosis`（`id` 传完整 64 位 `scope_hash`） |
| `ingest` | POST | `{source: "host"}` 或 `{source: "export", payload}` |
| `analyze` | POST | `{with_evaluation: bool}` |
| `report` | GET | 最近一次分析结果 |
| `policies` | GET | 策略记录 |
| `policy` | POST | `{version, action: accept\|ignore\|reopen\|rollback}` |
| `export` | GET | 导出样本、报告与策略 JSON |
| `reset` | POST | `{confirm: "reset"}` 清空本插件数据 |

---

## 隐私

- 样本从不包含消息正文，即使 ChatDynamics 打开了 `console_show_message_content`；
- 会话标识与消息 ID 只以哈希或首尾脱敏形式出现在控制台；
- 保存的决策轨迹是本体已经白名单化的 schema 2 快照，本身不含正文；
- 建议里不带用户级标识符。

跨插件读写面的完整说明见 [`docs/contract.md`](docs/contract.md)。

---

## 开发

```powershell
python -m pytest -q          # 196 项：契约、作用域、样本、指标、策略、学习者、评测、契约健康度、画像、插件面
python -m ruff check .
python -m mypy

# 端到端冒烟：合成标注 → 导入 → 分析 → 评测 → 策略，全程不需要 AstrBot
python scripts/smoke.py
python scripts/smoke.py --sessions 30 --per-session 24 --json out.json
```

`scripts/smoke.py` 会在内存里跑完整条链路并打印控制台会看到的内容，包括评测结论与
每一条建议的「可采纳 / 仅诊断」标记。样本量影响留出集，因此小批量更容易得到
「样本不足」或「拒绝」——那是门限在工作，不是故障。

代码结构：

```text
core/
  config.py            有界配置解析
  trace.py             schema 2 轨迹归一化（契约层）
  features.py          冻结的确定性特征向量
  scope.py             作用域身份：当前 = 会话身份（v0.6.1）
  samples.py           LearningSample 与标注转换
  candidates.py        候选集、召回与「生成缺失 / 排序错误」归因（v0.6）
  quality.py           数据契约健康度：能力矩阵 + 原始记录计数（v0.6.1）
  metrics.py           不臆造分母的监督指标
  policy.py            真实配置键、有界增量、参数化回放判定
  logistic.py          无依赖、确定性的逻辑回归与阈值扫描
  recommendation.py    可采纳建议 vs 工程诊断
  recipient_learner.py v0.2
  topic_learner.py     v0.3
  evaluator.py         v0.4 按会话切分的离线评测
  scope_profile.py     会话画像、leave-one-out 基线、平滑与诊断门禁（v0.7）
  report.py            分析快照组装
  ingest.py            只读契约读取
  store.py             分片、有界的持久化
  web_api.py           Web API
```

---

## 版本路线

| 版本 | 内容 | 状态 |
| --- | --- | --- |
| v0.1 | LearningSample / DecisionTrace / 标注转样本 / 统计 | ✅ |
| v0.2 | Recipient Learning | ✅ |
| v0.3 | Topic Learning | ✅ |
| v0.4 | 离线评测框架：baseline vs candidate | ✅ |
| v0.5 | 小步累计调参 + 错误类型归因 + 候选召回拆解 | ✅ |
| v0.6 | 候选归因修正（不存在 ≠ 空集合）+ 契约字段对齐 | ✅ |
| v0.6.1 | 作用域身份（scope）+ 数据契约健康度（能力矩阵） | ✅ |
| v0.7 | Scope Dynamics Profile：会话画像 + LOO 基线 + 诊断门禁 | ✅ |
| v0.8 | Participation Learning（什么时候该闭嘴） | 计划 |
| v0.9 | Group Adaptive Policy（需要本体先提供跨会话群身份） | 计划 |
| v1.0 | Semi Auto Tune + Safe Adaptive Dynamics | 计划 |

### 依赖本体的一项 P0

`routing.topic_candidates` 的**稳定写入**是 Topic Learning 上限的直接瓶颈：没有候选集，
就无法区分「embedding 没召回正确话题」和「召回了但规则选错了」——两者的修法完全相反。
本插件已经按结构化格式（`topic_id` / `final_score` / `rank` / `evidence`）写好了解析，
本体补齐后候选召回与错误归因会立刻生效。字段规格见
[`docs/contract.md`](docs/contract.md) 第 4b 节。

**不会先做 AI 模型训练。** 规则统计 + logistic regression + Bayesian prior + 阈值优化
在这个问题上已经足够，而且每一条结论都能被人工复核。

---

## License

MIT，与 ChatDynamics 一致。
