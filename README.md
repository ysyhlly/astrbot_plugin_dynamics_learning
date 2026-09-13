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
- 分别统计**对话对象**、**话题**、**回复准入**、**最终发送**四个任务的误差模式与证据倾向；
- 把每条被标注的消息归入**唯一一层**（错误归因链），回答「该改哪一层」而不只是「错了多少」；
- 用**两个留出集**（按会话 + 按时间前向）评测 baseline 与 candidate，并给出按会话重采样的
  95% 置信区间；区间跨 0 就不给可采纳结论；
- 在没有学习之前先过**数据门槛**：样本量、会话数、正负比例、轨迹降级率、候选与结果覆盖率、
  标注年龄、schema 分布；不合格就只出统计与诊断，不出策略；
- 把参数建议绑定到 ChatDynamics 的**真实配置键**，并按硬上限截断；
- 用 proposed → validated → shadow → promoted 状态机管理策略，只把 promoted 的记录经
  `GET /published` 发布出去，是否采用由本体决定。

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
| `learning_forward_holdout_ratio` | 0.25 | 按**标注时间**切分的前向留出比例（旧段训练、新段验证） |
| `learning_require_forward` | true | 是否要求前向验证也通过 |
| `learning_bootstrap_iterations` | 600 | 置信区间重采样次数（按会话成对） |
| `learning_bootstrap_seed` | 7 | 固定种子：同一批数据每次得到同一个区间 |
| `learning_bootstrap_alpha` | 0.05 | 0.05 = 95% 置信区间 |
| `learning_require_ci` | true | 区间跨 0 时是否拒绝 |
| `learning_group_min_support` | 12 | 会话分层诊断的最低支撑，低于它只列名字不给数值 |
| `learning_gate_min_samples` | 60 | 数据门槛：低于此数不出策略 |
| `learning_gate_min_sessions` | 4 | 数据门槛：会话太少切不出留出集 |
| `learning_gate_min_positive_rate` | 0.05 | 数据门槛：正类占比过低时降级为警示 |
| `learning_gate_max_degraded_ratio` | 0.5 | 数据门槛：轨迹降级比例上限（警示） |
| `learning_gate_max_label_age_days` | 120 | 数据门槛：标注最大年龄，超过就直接不出策略 |
| `learning_shadow_min_samples` | 500 | Shadow 门槛：保留窗口内去重后的有效比较数 |
| `learning_shadow_min_disagreements` | 100 | Shadow 门槛：分歧数，数量只是准入条件不是效果证据 |
| `learning_shadow_max_regression` | 0.01 | Shadow 门槛：总体准确率回退上限 |
| `learning_shadow_target_relative` | 0.10 | Shadow 门槛：目标错误率的相对改善 |
| `learning_shadow_target_absolute` | 0.01 | Shadow 门槛：目标错误率的绝对改善（与上一行二者其一） |
| `learning_shadow_relative_min_error` | 0.05 | Shadow 门槛：基线错误率低于此值不给相对改善结论 |
| `learning_shadow_ci_floor` | -0.002 | Shadow 门槛：变化量 95% 区间下界 |
| `learning_shadow_min_sessions` | 3 | Shadow 门槛：收益必须覆盖的会话数 |
| `learning_shadow_min_active_hours` | 4 | Shadow 门槛：必须覆盖的活跃时段（按本体判定时间的 UTC 小时） |
| `learning_shadow_subgroup_regression` | 0.05 | Shadow 门槛：单个分组的灾难性回退上限 |
| `learning_shadow_subgroup_support` | 20 | Shadow 门槛：子群回退判定所需支撑 |
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

### reply —— 此刻要不要接话，以及到底有没有接上

这是**两个问题**，所以是两个任务：

| 任务 | 预测目标 | 回答 |
| --- | --- | --- |
| `reply_admission` | `participation.level == "strong"` | 该不该进入回复流程 |
| `reply_outcome` | schema 3 的 `outcome.delivered` | 最后到底有没有发出去 |

拆开的原因是一个具体的误判：

   准入正确 + 被作息 / 降温 / 媒体压掉   → 以前算「漏回复」，其实是门禁干的，动阈值没用
   准入正确 + 生成超时                  → 同上，是生成阶段的问题
   准入错误                            → 这才是路由该背的

前两种在 schema 2 里长得和第三种一模一样（因为本体根本没记最终结果）。schema 3 记录
之后，`reply_outcome` 样本的 `error_type` 会直接写清停在哪一环：
`gate_suppression` / `generation_failure` / `delivery_failure` / `participation_error`。

**最终发送层不可回放**：门禁、生成与平台发送都不在记录轨迹里，所以它只报告已记录的
事实，不参与任何阈值回放，也不参与可采纳结论。

---

## 错误归因链：该改哪一层

只报「回复 F1 = 0.71」的语料没法告诉人该做什么。所以每条**被标注的消息**会被归入
唯一一格（不是每条样本 —— 一条消息最多产出四个样本，按样本数会让桶加起来超过语料规模）：

| 归因 | 该改哪里 |
| --- | --- |
| `recipient_error` | 定向：`strong_addressivity_threshold` 或定向证据 |
| `topic_candidate_miss` | 候选生成（embedding / 检索）：改阈值无用 |
| `topic_ranking_error` | 打分与排序：`topic_commit_threshold` / `topic_margin_threshold` |
| `participation_error` | 参与准入评分与阈值 |
| `gate_suppression` | **不是路由问题**：作息 / 降温 / 媒体把发送压掉了 |
| `generation_failure` | **不是路由问题**：生成阶段失败 |
| `delivery_failure` | **不是路由问题**：平台发送失败 |
| `unattributable` | 记录不足：需要本体补齐候选集或最终结果 |
| `ok` | 这条链上没有发现错误 |

三条性质是硬约束：

1. **桶是划分。** 每格一条消息，合计等于语料规模 —— 归因表不会悄悄变小；
2. **这是复查顺序，不是因果结论。** 四个层基本独立，一层错不会导致下一层错。同时错在
   两处的消息只记最先的一格，另一处记进 `also_failed`，不重复计数；
3. **没记结果 ≠ 没回复。** schema 2 消息如果准入是对的、结果未知，落在 `ok` 并带上
   `outcome_unavailable`，绝不落进失败桶里猜。

话题那一格复用 Topic Learner 自己的判定（`candidate_observations`），不二次推导：
两份实现会在同一条消息上给出两个都印着数字的答案。

## Shadow A/B：分歧子集才是结论所在

本体把 `learning_policy_mode` 设为 `shadow` 时，运行时**完全不变**，但每条消息会同时算出
「baseline 会怎么判」和「策略会怎么判」，记录进 schema 3 轨迹的 `shadow` 段。

真正要统计的不是总体准确率。阈值动几个百分点时，95% 以上的判定完全一样，总体变化是被
稀释之后的结果：

```text
10,000 条，分歧 320 条
总体变化     +0.5pp     同样的 320 条，摊到 10,000 条上
子集变化    +15.9pp     它在真正被影响的样本上的效果
```

所以评估建立在**配对表**上，四格划分全部带标签的消息：

```text
                 shadow 对        shadow 错
baseline 对      both_correct     baseline_only    ← 策略的代价
baseline 错      shadow_only      both_wrong
                 ↑ 策略的收益
```

两个非对角格正好就是分歧子集，所以 `changed == baseline_only + shadow_only` 恒成立
（`balanced` 字段自报，测试也钉住它 —— 一张加不起来的表会让所有从它推出来的数字失去意义）。

**「赢的没输的多」是最难被总体数字发现的一种失败**：40 条赢、60 条输、9,900 条两边都对时，
总体只动 −0.2pp，轻松通过 1% 的回退上限。只有配对表会说这个策略输了，所以 `net_gain`
是一条独立门槛。

### 进入 active 的门槛

| 检查 | 默认 | 为什么 |
| --- | --- | --- |
| `shadow_samples` | ≥ 500 条带标签 | 样本不足时下面每个比例都只是计数 |
| `disagreements` | ≥ 100 条分歧 | 分歧太少等于策略没被任何东西检验过 |
| `overall_regression` | ≥ −1% | 部署承受的是整个语料，不是子集 |
| `target_improvement` | 绝对 ≥ 1% 或相对错误率改善 ≥ 10% | 相对那一支要求基线错误率 ≥ 5%：0.3% → 0% 是 100% 的相对改善，也是 0.3pp，让这种数字满足「好 10%」会让规则在语料本来就好的地方失效 |
| `interval_floor` | 95% 区间下界 ≥ −0.2% | 下界低于这个值就是回退，不是「还没测出来」 |
| `sessions` / `active_hours` | ≥ 3 个会话 / ≥ 4 个时段 | 会话太少是在拟合一段对话；时段太少结论只对某个时间成立 |
| `subgroups` | 支撑 ≥ 20 的会话回退不得 > 5% | 全局变好不等于每个群变好 |
| `net_gain` | > 0 | 见上：赢的没输的多 |

活跃时段用的是**本体记录的决策时间**（`shadow.recorded_at`），不是样本时间戳 ——
后者是人工标注的时刻，一个晚上集中复查一周的消息会被算成同一个时段。

**当前作用域就是会话**，所以「子群」还不是真正的群身份；要按群做诊断，先等本体提供跨会话的
群标识。这一层的数字因此只作为诊断，不产生任何本地策略。

## 策略状态机与发布面

    proposed  →  validated  →  shadow  →  promoted  →  superseded / rolled_back

每一支箭头是一种不同的证据，把它们并起来就是「一个数字变成一次没人同意的配置变更」：

| 状态 | 意味着什么 |
| --- | --- |
| `proposed` | 学习者产出了它，什么都还没验 |
| `validated` | 它在自己没拟合过的留出集上赢了（**自动路径到此为止**） |
| `shadow` | 它被发布到线上行为旁边被观察过 |
| `promoted` | 人采纳了它；本体可以从 `/published` 读 |
| `superseded` / `rolled_back` | 被新版本取代 / 采纳后又被撤回 |

`validated → promoted` 是允许的，而且是**记下来**的：发布记录里带 `shadow_observed`，
说明这份策略到底有没有被看着跑过。要求必须点过 shadow 不会凭空造出影子证据，只会造出
一个为了走到下一步而点的状态。

跳过箭头的操作会被拒绝并说明两个状态名。`GET /published` 只发 `promoted` 的记录，
发的是**解析后的全部参数**而不是增量：只发增量的消费者得自己补全其余项，而那个补全
会静默变成它实际应用的值。

发布的文件里同时出现**三个版本号**，而且互不推导：

```json
{
  "policy_contract_version": 1,
  "policy_id": "policy_v3",
  "state": "promoted",
  "source": {
    "trace_schema_version": 3,
    "dataset_fingerprint": "…",
    "learning_version": "1.0.0"
  },
  "target": {
    "chat_dynamics_version": "v1.6.2",
    "baseline_config_hash": "3ab41f0c9d2e7b85",
    "validated_host_versions": ["v1.6.2"]
  },
  "params": { "...": "全部参数，不只是变动的那个" }
}
```

- `policy_contract_version` 是这个文件本身的协议版本（Learning → ChatDynamics）；
- `source.trace_schema_version` 是策略**训在哪种 trace 上**（ChatDynamics → Learning）；
- `target.chat_dynamics_version` 是它**对着哪个本体版本验证的**。

兼容性靠**验证出来的** `validated_host_versions`，不做 SemVer 推断 ——
`1.7.0 → 1.7.1` 可能改掉一个参与度计算或门禁顺序，而这份文件里每个阈值都是对着旧分布
校准的。列表今天只含训练时观测到的那一个版本；本体没报版本时为空，而空是「无法验证」，
不是「匹配」。本体侧：`active` 要求严格相等，`shadow` 允许 `version_mismatch` 但绝不实际应用。

`dataset_fingerprint` 是**来源证明**，不是运行时兼容条件：它必须在策略里、必须被记录和
展示，但不因为缺失或不同而拒绝。要「人为批准锁定」，锁 `policy_id` 才是对的那个东西。

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

判定规则：

- 留出集任一任务样本 < `learning_min_evaluation_samples` → **样本不足**，不下结论；
- 任一**核心指标**回退 > `learning_max_regression` → **拒绝**；
- 主指标（recipient 准确率 / topic 配对准确率 / 回复准入 F1）提升 ≥ `learning_min_improvement`
  → 进入下面的三重门槛；
- 否则 → **拒绝**。

### 采纳门槛：两个留出集 + 一个区间

   会话留出集   它不是在背某一段对话         （按 session 切）
   前向留出集   它没有在群改变习惯后失效      （按标注时间切：旧段训练、新段验证）
   置信区间     提升比重采样噪声大           （按会话成对 bootstrap，默认 95%）

任一条跑不起来时结论是**样本不足**，不是**可采纳**：「没看」和「看了没问题」不能给出
同一个结论，否则门槛就是装饰。自动路径最多到 `validated`；`promoted` 需要人点一下，
或者本体真的做过影子观察。

区间怎么读：

    主指标变化 +0.0080  95% CI [-0.0021, +0.0193]   ← 跨 0：与噪声无法区分
    主指标变化 +0.0140  95% CI [+0.0031, +0.0258]   ← 整段在 0 以上

重采样的单位是**会话**而不是样本：同一段对话里的两条消息共享话题、收件人历史和情绪，
按样本重采样会得到一个恰好窄掉「语料聚集程度」那么多的区间。

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
作用域身份、最终发送结果、候选逐条证据），带自己的可用条数、合计条数、覆盖率，以及
每一条被排除的原因。

「最终发送结果」这一行的分母是**所有带回复标注的消息**，分子是其中也记录了 schema 3
`outcome` 的那些。schema 2 数据下它是「不支持」——本体没写结果，所以「该回但被作息
压掉」和「该回而路由没回」在这些记录上是同一条记录。它不是错误，是数据回答不了的问题；
本体哪天开始写 `outcome`，它就会自动变成一项可用的能力。

同一张表下面还有**数据门槛**：那是在任何学习之前跑的检查，不通过就完全不产生策略记录
（连 `proposed` 都不记，否则控制台里会躺着一个等着被点的版本号）。

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
| 数据契约状态 | 能力矩阵、数据门槛（阻塞项 / 限定条件）、契约面计数与最近一次导入的字段缺口 |
| 学习概览 | 样本数、会话数、策略记录数、数据来源与上次导入诊断 |
| 错误归因链 | 每条消息归入哪一层、该改哪一层、同时出错与压制原因明细 |
| 最近 7 天标注表现 | 四个任务在窗口内的标注准确率（始终附样本数） |
| 错误分布 | 按任务与错误类型汇总 |
| 系统建议 | 参数建议（含 before→after、理由、置信度、评测结论）与工程诊断 |
| 离线评测 | baseline vs candidate 逐指标对比、95% 置信区间、采纳门槛（两个留出集）与按会话分层 |
| 环境层学习评分 | 拟合评分的留出集表现与它需要的前提（诊断，不可导出） |
| Shadow A/B | 配对表（两格一致 / 两格分歧）、分歧子集准确率、95% 区间与进入 active 的逐项门槛 |
| 策略版本 | 版本列表、状态机操作（标记已验证 / 影子观察 / 采纳 / 忽略 / 回滚）、验证结果与已发布日期 |
| 样本浏览 | 分页查看样本（无正文，身份字段脱敏） |
| 会话画像 | 会话列表（被检查样本量、置信度、主要问题、候选链诊断）与单个会话的偏差明细 |

## Web API

全部挂在 `/astrbot_plugin_dynamics_learning/` 下，走宿主 Dashboard 的插件鉴权：

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `overview` | GET | 状态、配置、数据面、上次导入诊断 |
| `samples` | GET | 分页样本（`page` / `page_size` / `task` / `session` / `scope`） |
| `quality` | GET | 数据契约健康度：能力矩阵 + 数据门槛 + 契约面计数（含 `contract_at`） |
| `attribution` | GET | 错误归因链（`examples` 控制样例条数） |
| `shadow` | GET | Shadow A/B：配对表、置信区间与进入 active 的门槛 |
| `scopes` | GET | 作用域列表：被检查样本量、置信度、主要问题 |
| `scope` | GET | 单个作用域画像：`profile` / `global`（LOO 基线）/ `deltas` / `diagnosis`（`id` 传完整 64 位 `scope_hash`） |
| `ingest` | POST | `{source: "host"}` 或 `{source: "export", payload}` |
| `analyze` | POST | `{with_evaluation: bool}` |
| `report` | GET | 最近一次分析结果 |
| `policies` | GET | 策略记录 |
| `policy` | POST | `{version, action: validate\|shadow\|accept\|ignore\|rollback\|reopen\|promote\|supersede}` |
| `published` | GET | **只读发布面**：只有 `promoted` 的策略，供 ChatDynamics 决定是否采用 |
| `export` | GET | 导出样本、报告、策略与已发布策略 JSON |
| `reset` | POST | `{confirm: "reset"}` 清空本插件数据 |

---

## 隐私

- 样本从不包含消息正文，即使 ChatDynamics 打开了 `console_show_message_content`；
- 会话标识与消息 ID 只以哈希或首尾脱敏形式出现在控制台；
- 保存的决策轨迹是本体已经白名单化的快照（schema 2 或 3），本身不含正文；
- 建议里不带用户级标识符。

跨插件读写面的完整说明见 [`docs/contract.md`](docs/contract.md)。

---

## 开发

```powershell
python -m pytest -q          # 268 项：契约、schema 2/3 读取、作用域、样本、指标、策略、学习者、
                             #  归因链、评测、前向验证、置信区间、数据门槛、契约健康度、画像、插件面
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
  outcome.py           最终结果与抑制原因：开放词表 + 阶段判定（v0.8）
  trace.py             schema 2 / 3 两个 reader + 归一化层（v0.8）
  features.py          冻结的确定性特征向量
  scope.py             作用域身份：当前 = 会话身份（v0.6.1）
  samples.py           LearningSample 与标注转换（四个任务）
  candidates.py        候选集、召回与「生成缺失 / 排序错误」归因（v0.6）
  buckets.py           错误归因桶的唯一词表（v0.8）
  attribution.py       消息级错误归因链（v0.8）
  shadow.py            Shadow A/B：分歧子集、配对表与 active 门槛（v1.0）
  quality.py           数据契约健康度：能力矩阵 + 数据门槛 + 原始记录计数
  metrics.py           不臆造分母的监督指标
  policy.py            真实配置键、有界增量、参数化回放判定、状态机与发布面
  bootstrap.py         按会话成对重采样的置信区间（v0.9）
  logistic.py          无依赖、确定性的逻辑回归与阈值扫描
  recommendation.py    可采纳建议 vs 工程诊断
  recipient_learner.py v0.2
  topic_learner.py     v0.3
  evaluator.py         按会话 / 按时间的双留出集评测、门槛与分层诊断
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
| v0.8 | Outcome Learning：schema 3 读取、双层回复、错误归因链 | ✅ |
| v0.9 | Forward Validation + Confidence：前向留出集、bootstrap 区间、数据门槛、分层诊断、策略状态机 | ✅ |
| v0.10 | Group Adaptive Policy（需要本体先提供跨会话群身份） | 计划 |
| v1.0 | Semi Auto Tune + Safe Adaptive Dynamics | 计划 |

### 依赖本体的两项 P0

1. **`routing.topic_candidates` 的结构化稳定写入**，带逐条 `evidence`。没有候选集就无法
   区分「embedding 没召回正确话题」和「召回了但规则选错了」——两者的修法完全相反；
   没有逐条证据就无法回答「是哪个分项把错的候选排到了前面」。
2. **`outcome` 的写入**（`final_outcome` / `delivered` / `suppression_reason` /
   `stage`）。没有它，「该回但被作息压掉」和「该回而路由没回」是同一条记录，
   回复层只能按路由准入解释。

两项的字段规格见 [`docs/contract.md`](docs/contract.md) 第 4 节。本插件已经把两个 reader
与归一化层写好，本体补齐后候选召回、逐条证据与最终发送结果会立刻生效 —— 在那之前，
schema 2 记录会被明确标成 `outcome_unavailable` 与 `candidate_evidence_partial`。

**不会先做 AI 模型训练。** 规则统计 + logistic regression + Bayesian prior + 阈值优化
在这个问题上已经足够，而且每一条结论都能被人工复核。

---

## License

MIT，与 ChatDynamics 一致。


### Operational Shadow Coverage

`/shadow` 的 `operational` 区块读取本体独立的 `shadow_telemetry_v1` 快照，统计保留窗口内真实比较、分歧率、policy_id + host_version 分桶，以及 reason、匿名 session 和 UTC 小时时段分布。它不会生成标注样本；四格表和净收益继续来自人工标签。读取失败和没有快照分别显示，不能将它们理解成零流量。

本体默认最多保留 20000 条、30 天，每 30 秒及正常退出写入快照。因此这是保留窗口分母，不是历史累计流量，异常退出可能丢失最近一个快照周期的数据。第一轮只改变 strong_addressivity_threshold；收集至少 2000 次有效比较、100 条已标注分歧并核对净收益、CI、核心指标及多会话覆盖，再评估晋升。
