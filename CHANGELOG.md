# Changelog

## v1.2.0 — 回复复盘：模型逐条判断「这条该不该回」

能力矩阵回答「这批数据能不能支撑某项分析」，那是关于语料的问题。这一版加上另一个问题，一条一条地问：
这条消息，机器人当时应该回复吗？需要的证据也不同 —— 判断一条消息需要那条消息本身，而本插件从来不保存正文。

### 模型看不到答案

- 新增 `core/reply_review.py`。发给模型的只有：正文、这条消息有没有提到机器人、以及匿名会话分组
  （同一段对话的消息能对上，但看不出是哪个群）。**不给人工标注、不给本体判定、不给实际结果** ——
  一个先看过答案的评审只会同意答案，所以每一条对照才有信息量。
- 判断回来后与三方并排：人工标注（`expected_reply`）、本体档位与实际结果（schema 3 结果段）、模型判断，
  外加两列对照 —— 模型 vs 实际（漏回 / 多回 / 一致）与模型 vs 人工（一致 / 不一致）。
- 没有结果记录的条目按路由档位对照并明确标注「结果未记录」；模型漏答的条目保留为「未判断」，
  编造的 `msg_id` 丢弃并列出。

### 正文只走一趟

- 正文只在本体共享首选项 → 本插件内存 → 模型之间过一次：不写进样本存储，也不写进复盘结果。
  复盘结果只活在插件进程内，重启即忘 —— 这也是它没有像契约解读那样缓存到插件 KV 的原因。
- `learning_reply_review_enabled` 默认关闭，关闭时**连本体都不读**（不是读出来不发），面板也只在点
  「开始复盘」时才调用模型。
- 本体没有正文（未打开「控制台显示消息正文」）时**不调用模型**：状态明确区分「本体读不到」、
  「本体还没有可复盘记录」与「选中条目都没有正文」三种情况，各自说明要做什么。

### 取哪几条

- 取最近 N 条可复盘消息（有人工标注、有路由档位或有结果记录），默认 12 条、最多 40 条，
  按标注时间倒序；统计里给出候选总数、带正文/带标注/带结果条数、涉及会话数与「标注与结果互相矛盾」的条数。
- 新增只读接口 `GET reply_review`（`?refresh=1` 重新复盘）。失败沿用与契约解读相同的 90 秒退避。

### 配置

- 新增 `learning_reply_review_enabled`（默认 false）、`learning_reply_review_provider`（留空用宿主当前对话模型）、
  `learning_reply_review_timeout`（默认 60 秒）、`learning_reply_review_messages`（默认 12，范围 1~40）。

### 验证

- 新增 `tests/test_reply_review.py`（22 项）：摘要必须不含标注/判定/结果/会话标识、批次挑选与统计、
  三方对照的四种结论、未记录结果时退回档位对照、漏答与编造 id、非布尔判断、置信度截断、围栏与纯文本回复、
  关闭时连本体都不读、正文到达模型但不进存储、缓存与退避、宿主不可读与无正文两种状态。
## v1.1.0 — 数据契约状态改由模型解读

「数据契约状态」以前完全由本插件算出来。它的每一格都能对上语料，却回答不了持有数据的人真正要问的
问题：这批数据现在能做什么、下一步先补什么。这一版把这一页交给模型重写，同时把三件事钉死在插件里。

### 模型能看到什么

- 新增 `core/review.py`：`build_digest` 按**白名单**从质量载荷里摘出模型可以看的事实 —— 数据集计数、
  轨迹 schema 分布、每条能力的 id / 定义 / 确定性状态 / 可用数 / 合计 / 覆盖率 / 原因、契约面发现、
  数据门槛。载荷以后新增字段不会自动跟着发给模型；摘要里没有消息正文，也没有任何身份信息。
- `digest_fingerprint` 是摘要的稳定标识：刷新页面、重启进程都不会让同一批数据被反复解读，
  数据或契约面一变指纹就变，缓存自然失效。

### 模型能说什么

- 模型改写整张表：每条能力的状态、短标签、说明与「下一步」，以及面板级的一句话结论、verdict、
  动作清单与前提。它可以不同意确定性判定，改判行会标成「模型改判」并保留插件原本的状态。
- 模型漏掉的能力不会消失：按确定性判定补齐并标成「原始判定」，表格永远完整。
- 状态语汇与 `core/quality.py` 共用一份，不认识的取值会换回确定性状态；编造的能力 id 会被丢弃
  并在面板上说明丢了哪些。

### 数字不由模型负责

- 面板渲染的可用数、合计与覆盖率取自摘要而非模型回复 —— 模型决定这一行「是什么意思」，
  语料决定它「是多少」。
- 回复里出现、摘要中不存在的数字会被单独列出（`unverified_numbers`）并显示成一条警示。
  比例的百分比写法（0.8 → 80%）算同一个数；数量的百倍写法不算（20 → 2000 会被拦下）。

### 失败是常态

- 关闭、宿主没有可用 Provider、超时、回复不是可解析的 JSON，每一种都返回同样的结构并带上原因，
  面板回落到本插件判定，不会空白。失败后退避 90 秒：刚失败的 Provider 不会因页面刷新被反复调用，
  只有 `?refresh=1`（面板上的「重新解读」）才立刻重试。
- 确定性表格没有删除：折叠在模型面板下方「原始判定与计数」，模型不可用时它就是面板本身。

### 配置与接口

- 新增 `learning_review_enabled`（默认开启）、`learning_review_provider`（留空用宿主当前对话模型）、
  `learning_review_timeout`（默认 45 秒）。
- 新增只读接口 `GET review`（`?refresh=1` 强制重算），结果按摘要指纹缓存在插件 KV。

### 验证

- 新增 `tests/test_review.py`（17 项）：摘要白名单与指纹、提示词、合法/围栏/纯文本回复、改判标记、
  未知状态回落、编造能力、越界数字拦截、缓存与退避、无 Provider、Provider 抛错、端点注册。
## v1.0.0 — Shadow A/B：分歧子集评估与进入 active 的门槛

- 独立 candidate API/KV 提供 validated/shadow 策略用于真实影子观察，published 仍只提供 promoted。
- 增加 Operational Shadow Coverage：读取独立无正文 telemetry，去重后按策略、本体版本、原因、会话与时段统计，保留窗口分母与人工标签收益分开显示。
- 数据契约状态按数据说话：能力行的「不支持」不再一律解释成 schema 2 的契约缺口 —— 没有标注样本时只说还没有样本，轨迹声明 schema 2 时才说契约不写 outcome，声明 schema 3 却没有 outcome 段时按**记录缺失**报出（`contract_findings` 同样区分）；面板另显示本体实际写入的 trace schema 分布与读取端支持的版本。

这一版把闭环补上：本体在 `shadow` 模式下**不改行为**，但每条消息会同时算出
「baseline 会怎么判」与「策略会怎么判」，学习层据此回答唯一值得问的问题 ——
**策略在它影响的那些样本上，是不是比 baseline 更对。**

### 为什么不是看总体准确率

阈值动几个百分点时 95% 以上的判定完全一样，所以总体变化是策略效果被稀释之后的结果：

```text
10,000 条，分歧 320 条
总体变化   +0.5pp     同样的 320 条，摊到 10,000 条上
子集变化  +15.9pp     它在真正被影响的样本上的效果
```

新增 `core/shadow.py`：`shadow_rows`（按**消息**分组，不是按样本 —— 一条消息最多产出四个样本，
按样本读会把同一次比较最多算四遍）、`disagreement_table`、`evaluate_shadow`。

### 配对表是四格划分，而且会自报守恒

```text
                 shadow 对        shadow 错
baseline 对      both_correct     baseline_only    ← 策略的代价
baseline 错      shadow_only      both_wrong
                 ↑ 策略的收益
```

两个非对角格正好就是分歧子集，因此 `changed == baseline_only + shadow_only` 恒成立。
`balanced` 字段自报这个恒等式，测试也钉住它 —— 一张加不起来的表会让所有从它推出来的
数字失去意义。写这一版时发现第一版实现把 `both_correct` / `both_wrong` 算在**分歧子集**
上，而二元判定下那两格在分歧子集里恒为 0：现在它们算在全部带标签消息上，是真正的划分。

### 进入 active 的门槛

八项检查，每项都自报测量值（只报失败的门槛读不出「没有失败」和「什么都没跑」的区别）：
样本量、分歧量、总体回退、目标改善、区间下界、会话数、活跃时段、子群灾难性回退，
外加一条 `net_gain > 0`。

`net_gain` 单独成一条，是因为「赢的没输的多」是最难被总体数字发现的失败：40 条赢、
60 条输、9,900 条两边都对时，总体只动 −0.2pp，轻松通过 1% 的回退上限。

相对错误率那一支加了一个下限：基线错误率低于 5% 时不给相对改善——0.3% → 0% 是 100% 的
相对改善，也是 0.3pp，让这种数字满足「好 10%」会让规则在语料本来就好的地方失效。

### 活跃时段用决策时间，不是标注时间

`shadow.recorded_at` 是本体做判定时的墙钟时间；样本时间戳是**人工标注**的时刻。
一个晚上集中复查一周的消息，按标注时间会被算成同一个时段。

### 契约与接口

- schema 3 轨迹新增 `shadow` 段（`policy_id` / 两个阈值 / 两个判定 / `changed` /
  `score` / 两侧 margin / `reason`）；`reason` 说明两边为什么一致
  （`structural` / `early_return` / `ambient`）；
- `EntityTrace.shadow` 与 `LearningSample.shadow`，与其他 schema 3 事实一样写在每条样本上；
- 契约面新增 `shadow.present` / `shadow.absent` 计数：本体对**每条**处理过的消息都记录了
  比较，但只有人工标注过的那些会成为样本，这两个计数是防止把 320 读成总体规模；
- 新增 `GET /shadow`；控制台新增「Shadow A/B：分歧子集」卡片；
- 11 个新配置键（`learning_shadow_*`）列入 `_conf_schema.json`，把门槛全部参数化 —— 一个没人能移动的门槛，就是一个没人能解释为什么改它的门槛。

### 跨仓库验证

`tests/test_cross_repo_contract.py` 新增两条：本体算出的 shadow 判定经真实
`build_routing_trace` 冻结、被学习层读成一次分歧；以及结构化轮次在两边都被认成
「同一判定」（`reason=structural`、`changed=false`）—— 两边的 admission 规则必须是同一件事，
否则分歧子集在契约两侧会是两个不同的集合。

## v0.9.0 — 归因链、双层回复与前向验证

这一版回答的问题从「还能算什么指标」变成「**该改哪一层**」。计划里把 Learning 侧拆成
v0.7.0（Outcome Learning）与 v0.8.0（Forward Validation + Confidence）两步；本仓库的
v0.7.0 已经是会话画像，因此两项合并在 v0.9.0 交付。

### P0-1 支持 ChatDynamics schema v3

`core/trace.py` 现在有两个 reader 加一层归一化，而不是「有什么读什么」：

| reader | 触发条件 |
| --- | --- |
| `read_schema_v2` | `trace_schema_version == 2` |
| `read_schema_v3` | `trace_schema_version == 3` |
| `normalize_trace` | 按记录的版本分发；版本缺失或不认识时按 schema 2 字段集读取并置 `degraded` |

新版正式消费 `routing.topic_candidates`（含逐条 `evidence`）、`routing.selected_topic`
与 `outcome.{final_outcome, delivered, suppression_reason, stage}`。两条规则：

1. **自描述字段照读。** `routing` / `outcome` 的键名本身说明含义，即使版本号没跟上也会
   被读取 —— 丢掉本体真写过的事实比版本不符更糟；
2. **版本决定证据等级。** 只有声明 schema 3 的记录才可能达到 `candidate_evidence = full`。

写回时写的是**读到的那个版本**，所以 schema 2 样本往返后仍是 schema 2，不会凭空长出
本体没写过的字段。

### P0-2 两个显式降级标记

schema 2 记录被明确标成 `outcome_unavailable` 与 `candidate_evidence_partial`，
而不是静默留空。`candidate_evidence` 是三值（`full` / `partial` / `none`）：
`none` 比 `partial` 更降级（连候选集都没有），两者都让 partial 标记为真。

抑制原因是**开放词表**：本体新增一个 `reason_code` 时不会被归进「门禁压制」，而是记成
`unknown` 并计数 —— 加一个原因应当表现为一个未分类的码，而不是一次静默的门禁压制。

### P0-3 回复学习拆成两层

| 任务 | 预测目标 | 回答 |
| --- | --- | --- |
| `reply_admission` | `participation.level == strong` | 该不该进入回复流程 |
| `reply_outcome` | `outcome.delivered` | 最后到底有没有发出去 |

拆开的原因是一个具体的误判：**准入正确但被作息 / 降温压掉的发送，以前和真正的路由漏回复
长得一模一样**，因为 schema 2 里根本没有最终结果。现在：

- 没有结果就**不产生** `reply_outcome` 样本（schema 2 数据下该任务为空，而不是一列编出来的
  `silent`）；
- 最终发送层**不可回放**：门禁、生成与平台发送都不在轨迹里，它只报告事实，不参与阈值回放
  与可采纳结论；
- 同名错误不再复用：准入层漏回复叫 `missed_reply`，发送层漏发送叫 `undelivered_reply`。

旧样本的 `task == "reply"` 按准入任务加载（它一直都是准入问题），`sample_id` 保持不变。

### P0-4 完整错误归因链

`core/attribution.py`：每条**被标注的消息**归入唯一一格，合计等于语料规模。

    recipient_error / topic_candidate_miss / topic_ranking_error / participation_error
    gate_suppression / generation_failure / delivery_failure / unattributable / ok

三条硬约束：桶是**划分**（不会悄悄变小）；这是**复查顺序而不是因果结论**（同时错在两处的
消息只记最先一格，另一处进 `also_failed`）；**没记结果 ≠ 没回复**（落在 `ok` 并带
`outcome_unavailable`，绝不进失败桶）。话题那一格复用 `candidate_observations` 自己的
判定，避免两份实现印出两个都带数字的答案。

### P1 前向验证、置信区间、数据门槛、分层诊断

- **时间前向留出集**：按标注时间切，旧段训练、新段验证。会话留出集问「换到没见过的会话
  还成立吗」，前向留出集问「换到更晚的数据还成立吗」，**两个都要过**；
- **按会话成对 bootstrap**：输出 `+0.8% 95% CI [-0.2%, +1.9%]` 而不是一个点估计；
  区间跨 0 即判定「与噪声无法区分」，不给可采纳结论。重采样单位是会话而不是样本 ——
  同一段对话的两条消息共享话题与情绪，按样本重采样会得到一个恰好窄掉聚集程度的区间；
- **数据门槛**在任何学习之前跑：样本量、会话数、标注年龄是阻塞项；正负比例、轨迹降级率、
  候选覆盖率、结果覆盖率是限定条件（它们只限定对应方向，不该一票否决另一个方向）。
  不通过就**完全不产生策略记录**，连 `proposed` 都不记 —— 否则控制台里会躺着一个
  等着被点的版本号；
- **按会话分层诊断**：指出全局变好时哪些会话反而变差了，并且**不产生本地策略**：
  在一个会话上拟合出的阈值是记住对话，不是学会相处。

### P1.5 策略状态机与发布面

    proposed → validated → shadow → promoted → superseded / rolled_back

自动路径最多到 `validated`：评测能证明离线提升，证明不了它没见过的门禁、生成与平台。
`validated → promoted` 允许，但**记下来**：发布记录带 `shadow_observed`，说明这份策略
到底有没有被看着跑过。跳过箭头会被拒绝并说明两个状态名。

`GET /published` 只发 `promoted` 的记录，发的是**解析后的全部参数**而不是增量：
只发增量的消费者得自己补全其余项，而那个补全会静默变成它实际应用的值。

策略记录新增 `training_dataset` / `holdout_result` / `forward_result` / `target_error` /
`collateral_regressions` / `confidence` / `compatibility` / `status_history`。
旧状态词按语义映射：`candidate → proposed`，`accepted → validated`（旧 `accepted`
是评测结论，从未发布给本体，因此不是 `promoted`）。

### 界面

新增「错误归因链」卡片；离线评测区增加置信区间列、采纳门槛块与按会话分层表；数据契约区
增加数据门槛表；策略版本表增加验证结果列与状态机操作按钮。样本浏览的任务筛选改为
`recipient` / `topic` / `reply_admission` / `reply_outcome`。

### 契约命名：三个版本号，互不推导

之前有个命名事故：`contract_version` 同时表示「本体写的 trace schema」和「本插件读取端的
修订号」，读一次就要猜 4 指的是哪一个。现在拆开：

| 名字 | 含义 | 谁改它 |
| --- | --- | --- |
| `trace.supported` / `trace.latest` | 本插件**能读**哪些 trace schema | 只有可读集合变化时 |
| `trace.observed` | 本体**实际写了**哪些（分布） | 只有本体 |
| `reader_version` | 本插件的读取端修订号（重置为 1） | 本插件 |
| `policy_contract_version` | `/published` 的协议版本（= 1） | 本插件 |

轨迹里的 schema 号也改由 `trace_schema_version` 承载：`routing_schema_version` 描述的是
routing 段，而数字描述的是整条轨迹。旧键仍然读取（已落库的标注不能因此读成「未记录」），
写回时只写新键。

### 发布面

`/published` 改为四段式，三个版本号各自归位：

```json
{
  "policy_contract_version": 1,
  "policy_id": "policy_v3",
  "state": "promoted",
  "source": { "trace_schema_version": 3, "dataset_fingerprint": "…", "learning_version": "0.9.0" },
  "target": { "chat_dynamics_version": "v1.6.2", "baseline_config_hash": "…",
              "validated_host_versions": ["v1.6.2"] },
  "params": { "…": "全部参数" }
}
```

- `dataset_fingerprint` 降级为**来源证明**：必须存在、必须记录与展示，但不再作为拒绝条件。
  只有消费端显式配置 `expected_dataset_fingerprint` 时才做 pin 校验；真正适合「人为批准
  锁定」的是 `policy_id`；
- 兼容性靠**验证出来的** `validated_host_versions`，不做 SemVer 推断：1.7.0 → 1.7.1 可能
  改掉一个参与度计算或门禁顺序，而这份文件里每个阈值都是对着旧分布校准的。列表今天只含
  训练时观测到的那一个版本，本体没报版本时为空 —— 空列表是「无法验证」，不是「匹配」；
- `active` 要求严格版本相等，`shadow` 允许 `version_mismatch` 但**绝不实际应用**（这条由
  本体实现）。

### 发布契约落成一个键

`/published` 之外，发布契约现在也**物化**在共享首选项的 `learning_published_v1` 里
（每次策略状态变化与每次分析结束时写入）。消费端读一个键就拿到商定好的契约，
而不是自己从策略记录里重新推导一遍形状 —— 同一份契约两份实现，就是两次走样的机会，
而走样的表现是「策略悄悄不生效」。清空存储时这个键一并删除。

### 跨仓库契约测试

新增 `tests/test_cross_repo_contract.py`：调用**真实**的 ChatDynamics
`build_routing_trace`、`outcome_recorder` 与 `learning_policy`，覆盖完整闭环 ——
本体写轨迹 → 学习层导入训练验证 → 产出 promoted 策略 → 发布 → 本体解析并确认
would-override 与学习层发布的**逐键相同**。本体的包不在导入路径上时测试会 skip，
而不是退回 fixture —— 一个会悄悄退化成 fixture 的跨仓库检查，正是它存在的理由的反面。

这个测试当场抓到了一处真实的跨仓库耦合：本体的 `topic_commit_threshold` 配置字段默认
0.0（由 `topic_join_threshold` 推导），而学习层的 `BASE_POLICY` 认为它是 0.58。
不修的话每一份发布摘要都对不上，`active` 永远不可达，而且看起来像是「运营改过配置」。

## v0.7.0 — 会话画像：先说清楚它不是什么

画像层回答的是「**在这个会话里，我人工检查过的那批消息，系统经常错在哪里**」。
它不回答「这个会话有多少比例的消息会出错」，而且这个区别写在每一段输出里——
标注是人挑出来标的，一个被专门翻查问题的会话天然比没人看的会话"更差"，
把选中样本的分布读成总体错误率，就是把"我在哪看"变成"它在哪差"。

### 对照的是 leave-one-out 基线，不是全局

偏差算的是「本会话」对「**除本会话以外的**全部样本」：

```text
A: 8/10 漏识别    B: 1/10    C: 1/10

含自己的全局   10/30 = 33.3%   → 偏差 +46.7pp（会话越大越接近和自己比）
leave-one-out   2/20 = 10.0%   → 偏差 +70.0pp
```

这不是精度问题，是方向问题：用包含自己的全局做分母，会把差异系统性朝 0 拉。
平滑用的先验也必须是同一条基线——本会话自己的错误不该进自己被收缩到的那个值。

### 小样本会被收缩，而且不装成特征

Beta-Binomial：`smoothed = (errors + 20 × loo_rate) / (support + 20)`。
三条样本里的两个错误不会变成 66.7% 的"群性格"，而且样本不足 `MIN_RATE_SUPPORT` 时
偏差直接不给——原始值、平滑值、基线值三个数一起给，偏差用平滑值算，
所以任何一次数字变化都能解释。

置信度分四档（样本不足 / 低 / 中 / 稳定），"稳定"要 100 条以上**且跨 3 个标注日**。
这里修掉一个自己埋的坑：原方案还要求「≥3 个会话」，但当前契约下一个作用域就是一个会话，
那条门禁永远不可能通过——通不过的门禁不是保护，是装饰。所以它跟着
`core/scope.py` 的 `SCOPE_SPANS_SESSIONS` 缩放：本体哪天提供真正的跨会话群身份，
它会自动收紧，画像代码不需要改。

时间维度用的是**标注时间**（`annotated_at`），不是消息发生时间。本体没有冻结消息时间，
所以它只能说明"这次复查是分散的"，不能说"这个会话长期如何"。

### 诊断先过门禁，再下结论

候选覆盖率 ≥ 70%、可算召回 ≥ 20 条、可算选中准确率 ≥ 20 条，三者之一不满足就输出
「候选证据不足」，而不是在十个样本上猜是 embedding 还是阈值的问题。过关后：

```text
Recall@3 低 + 选中准确率高  → candidate_generation   （改检索，暂不建议动 topic_commit_threshold）
Recall@3 高 + 选中准确率低  → ranking_or_scoring     （阈值与间隔才是可动的旋钮）
```

话题类错误的分母是**会话内配对**，不是样本数：四个已标注消息构成 6 个配对，
拿配对计数除以样本计数会得到一个没人定义过的比率。跨会话的配对更是永远不做——
两个会话里各有一个同名话题，不该被读成一次误合并。

### 界面

页面分成两个视图：**总览**（学习、建议、调参、评测）与**会话画像**（会话列表 + 单个
会话的偏差明细与诊断）。偏差、基线、诊断全部由 Python 算好再返回：服务端算过的数字
能测，页面自己重算一遍的基线只会在某天和旁边的报告不一致。

调参边界没有动：v0.7 只观察、只对比、只诊断，仍然没有任何 group-specific 参数，
`run_tuning()` 依旧只放行 recipient/topic 两个全局任务。

新增 `GET /scopes` 与 `GET /scope?id=<64 位 scope_hash>`；`samples` 增加 `scope` 过滤。

验证：`pytest` 196 passed、`ruff` clean、`mypy` 18 个模块零问题、`scripts/smoke.py` exit 0。

## v0.6.1 — 先补地基：作用域身份 + 数据契约健康度

### 作用域 = 会话，而且这一点必须写进代码

本体在正常事件路径下是 `session_key = unified_msg_origin or group_id`、
`umo = unified_msg_origin or session_key`，恢复逻辑还要求 `umo == session_key`——
所以 **`umo` 永远等于会话键**，它证明不了"两个会话是同一个群"。而 `group_id` 更不能用：
它不带平台前缀（本体自己的测试里两个适配器共用 `room`），而且在事件没有群号时会被
写成会话键本身，同一个真实群的 `group_id` 会随创建路径变化。

因此 `core/scope.py` 把作用域定义为会话，并把三条约束钉成注释与测试：

```text
scope_hash == session_hash(session_key)      # 逐字节，升级不变量
umo == session_key 时不重新哈希              # 否则旧样本会被劈成两个身份
group_id 只做诊断（group_hint_hash）         # 不得作为聚合键
```

换前缀重新哈希（`sha256("group:" + key)`）会把每一份已存样本劈成两个身份——
这正是这个模块存在的理由，也是它唯一的测试重点。`SAMPLE_SCHEMA_VERSION` 从死代码
变成真的：它原来只有定义没人写没人读，现在每行样本都带着 `sample_schema_version`。

### 契约面：在原始记录上数，因为样本层是有损的

新增 `GET /quality`，把「本体到底写了什么」和「学习层能用这些样本做什么」分开报告：

| 原始记录里的事实 | 归一化之后 |
| --- | --- |
| `routing_schema_version = 1` | 永远写回 `2`，从样本里数会得到 100% schema 2 |
| `contribution_total = null` | 永远读回 `0.0`，从样本里数分不出"判了 0 分"和"没判分" |

所以契约面只在导入时对原始记录计数并缓存（带 `contract_at`），样本面每次请求重算。
第二个坑顺手补上了：`contribution_total` 缺失会让阈值回放把一个从未存在的 0 分当成
决定性分数，于是样本层多了一个 `contribution_total_recorded` 标记——
和 v0.6.0 给候选集加的 `topic_candidates_recorded` 同构。

计数满足会计恒等式 `seen == kept + malformed + unknown_session`，并由 `balanced` 自报；
不守恒时它先说自己是坏的。字段类计数只统计进入样本的那部分记录，被丢掉的行由会计类
计数负责——分子的分母不能来自另一个人群。

健康度输出的是**能力矩阵**而不是一个"82 分"：每一项能力带自己的可用条数、合计条数、
覆盖率，以及每一条被排除的原因（结构化直判、无前置机器人消息、没记加性分数……）。
判据直接取自学习器自己的谓词（`policy.decide` 的分支、`topic_learner.replay_can_move`），
不在健康度模块里二次推导，否则两边迟早会各说各话。

顺带修掉一个静默空结果：`samples?session=12 位十六进制` 原来会被当成裸会话键再哈希一次，
返回 HTTP 200 + 0 行，看起来像"这个会话没有样本"。现在 400 并说明要用完整 64 位。

读取面记为 `contract_version = 3`（v0.6.1 起消费 `panel_runtime_v1` 的身份字段）。

验证：`pytest` 173 passed、`ruff` clean、`mypy` 17 个模块零问题、`scripts/smoke.py` exit 0。

## v0.6.0 — 候选归因接上 P1

`core/candidates.py` 之前只有一半是对的。它已经会算 `Candidate Recall@K`、
`Selection Accuracy` 和三种归因，但有一个语义错误贯穿到底：

```python
has_record=bool(candidates)      # 空列表 == 没记录
```

本体「找过，一个都没提出来」和「根本没写这个字段」被当成了同一件事。后果是每一条无法
归因的样本都被记成 `not_recorded`，真正的候选生成缺失被系统性低估——而分开这两种错误
正是这件事的全部意义。更早的一环让它在观测层已经无法补救：`samples.py` 把 `None` 和
`[]` 都归一成 `[]` 再写进 trace，出库时早分不出来了。

修法是把「本体有没有写这个字段」当成一个独立事实存下来：

```python
trace["topic_candidates"] = [...]            # 归一化后的 payload
trace["topic_candidates_recorded"] = True    # 本体写了这个字段，哪怕是空列表
```

旧样本没有这个标记，它们保持原来的读法（空 = 未记录），**不会**被追溯判成候选生成失败：
记录说不清楚的事情，不要替它下结论。

其余对齐 P1 的改动：

- 归因桶补齐为六个，新增 `new_topic_expected` 与 `unattributable`。每条话题标注恰好落入
  一个桶，归因表合计等于样本总数——原来 `NEW` 单例和无标签样本是被直接跳过的，报表会
  因此比语料小一圈。
- 无法解析的候选条目开始**计数**（`dropped_entries`），不再静默改变名次。
- 报告输出观测到的候选长度分布，让「本体只记前 3 个、所以 Recall@5 与 Recall@3 必然
  相同」这件事可见，而不是让一个被截断过的数字读起来像测出来的。
- `candidate_recall` 增加 `missed` / `not_recorded` / `excluded_new_topic`。

顺带修掉一个崩溃：`_candidate_notes` 在 `recall_at_3` 有值、而 `selection.accuracy` 为
`None` 时会对 `None` 做 `:.1%` 格式化并抛 `TypeError`。触发条件是「记录了候选集，但正确
话题一次都没进过候选集」——也就是候选生成最差的那一批，恰好是最需要看到报告的那一批。
现在显示「无法计算」。

验证：`pytest` 135 passed、`ruff` clean、`mypy` 15 个模块零问题、`scripts/smoke.py` exit 0。

## v0.5.1 — 宿主 SDK 改为可选依赖

修复一个只在**没装 AstrBot 的环境**里才暴露的类型检查缺陷。`core/ingest.py` 用
`from astrbot.core import sp` 读取宿主的共享首选项，配的忽略码是 `import-untyped`：
宿主装上时这个码是对的，没装时 mypy 报的是 `import-not-found`，忽略码不匹配，于是
`mypy` 在这个 import 上同时报出"找不到模块"和"忽略码没用上"两条错误。

两个码无法同时满足——`warn_unused_ignores` 会把另一个环境下用不到的那个判为错误。
所以不再写忽略码，改为**按名字在调用时解析**：

```python
module = getattr(importlib.import_module("astrbot.core"), "sp", None)
```

运行行为不变：宿主不存在时依旧降级成诊断并返回空结果，绝不抛异常。区别只是现在
无论宿主装没装，类型检查结果一致。

补了两个回归测试，**直接调用真实的 `collect_from_host`**（原来只有一个 monkeypatch，
真实回退路径没被覆盖过），锁住"宿主缺失不抛异常"和"拿到了不是宿主的东西就不半途使用"
两条契约。

验证：`pytest` 127 passed、`ruff` clean、`mypy` 15 个模块零问题、`scripts/smoke.py` exit 0。

## v0.5.0 — 小步累计调参与错误归因

### 调参规则重做

原方案要求一次 ±5% 的参数变化单独带来 ≥2% 提升。这在真实群聊噪声下基本不可能成立，
learner 会退化成"只学习、不采纳"。改为**小步累计**：预算花在"一次迭代能走多远"，
而不是"一次能跳多远"。

```text
单步变化上限    ±5%   相对于当前值
累计漂移上限    ±15%  相对于原始基线
最大连续步数    3     （0.95^3 = -14.3%，与累计上限自然吻合）
```

三级信任：

| 级别 | 条件 | 含义 |
| --- | --- | --- |
| Safe candidate | 累计 ≥ -0.2% 且目标错误改善 且无核心指标回退 | 采纳本步，可以再走一步 |
| Advance | 边际 ≥ +0.5%，或目标错误相对下降 ≥5% | 值得再走一步 |
| Promote | 累计 ≥ +1.0%，或目标错误相对下降 ≥10%（且累计 ≥ -0.2%） | 可进入 learned policy |
| Strong promote | 累计 ≥ +2.0% | 高置信度 |

止损：

- 连续两步边际收益 < +0.2% → 停止，保留已获得的收益；
- 任一步核心指标回退 > 1% → 停止并整轮回滚到基线；
- 累计漂移触及 ±15% 且仍在改善 → 停止并要求人工确认。

**2% 没有浪费，只是角色变了**：从"每次采纳的硬门槛"变成"非常确定值得升级"的强信号。

### 按错误类型评判，而不只看总准确率

```python
EvaluationResult(
    global_delta=+0.006,             # 总指标只动了 0.6%
    target_error="fragmentation",    # 这次调整针对的错误
    target_error_delta=-0.375,       # 该错误相对下降 37.5%
    collateral={"wrong_merge": +0.14},
)
```

误拆分从 40 降到 25 而总准确率只动 0.6%，是成功，不是无效。现在每一次调整都会
**指明它针对的错误类型**（``missed_bot`` / ``false_bot`` / ``fragmentation`` /
``wrong_merge`` / ``missed_reply`` / ``premature_reply``），并按该错误的相对变化
判断是否继续、是否采纳。其他错误作为 collateral 一并报告。

### 话题候选归因

一个错误的话题决策其实是两种：

- 正确话题**没进候选集** → 候选生成（embedding / 检索）的问题，改阈值没用；
- 正确话题**进了候选集但没被选中** → 打分 / 排序的问题。

现在拆成两个互不掩盖的指标：

- ``Candidate Recall@1/3/5``：正确话题是否出现在候选集中；
- ``Selection Accuracy``：**在正确话题已进入候选集的样本上**，宿主选中的比例。

以及逐条归因：``candidate_miss`` / ``ranking_error`` / ``not_recorded`` / ``correct``。

候选记录同时接受旧版 ``[[score, topic_id], ...]`` 与结构化
``[{topic_id, final_score, evidence, rank}, ...]``。旧格式没有显式排名，因此按分数
排序而不是按书写顺序；结构化格式里的 ``rank`` 优先。**没有分数的候选只计入召回，
不参与阈值回放**——把缺失的分数当成 0 会凭空发明一个宿主从未有过的理由。

### 其他

- ``Advance`` 与 ``Promote`` 的判定全部基于留出集，训练集只用来选参数。
- 参数发生位移但留出集上没有任何指标变化时，结论是「无变化」而不是「候选」——
  后者会声称一次收益为零的验证通过。
- 控制台新增「迭代调参」与「话题候选归因」两块。

## v0.4.0 — 离线评测框架

- 按**会话**切分训练集与留出集，拟合与阈值扫描只在训练集上做，留出集只被打分一次。
- 明确区分**可导出候选**（加性分数上的有界阈值移动）与**诊断评分**（拟合的环境层
  评分，需要本体开放「环境层评分替换」才能生效）。
- 策略版本记录：``policy_vN``、状态流转与评测结论一起持久化。
- precision 与 recall 报告但不参与守卫：它们天然此消彼长，同时守卫会否决任何平衡的
  调整；F1 已在守卫里。

## v0.3.0 — Topic Learning

- 会话内配对指标（``wrong_merge`` / ``fragmentation``），对标签重命名不变。
- 阈值回放：收紧可完整回放；放宽需要候选集，缺失的消息保持原归属，是下界而非估计。
- 误拆分占主导时给方向并指名本体需要补的字段，不产出无法验证的数字。

## v0.2.0 — Recipient Learning

- 人工样本准确率与混淆矩阵、错误类型分布、按证据码的贝叶斯平滑 lift、
  确定性逻辑回归的样本内权重、记录分数上的阈值扫描。
- 区分结构化轮次、无前序 Bot 消息的提前返回、以及真正参与评分的环境层轮次。

## v0.1.0 — 数据基础

- ``LearningSample``：一任务一样本，保存特征而不只是结论。
- ``DecisionTrace``：schema 2 决策轨迹的归一化，畸形输入降级而不抛异常。
- 只读读取共享首选项；按会话分片的样本存储；不保存消息正文。
