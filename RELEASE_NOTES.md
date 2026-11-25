# astrbot_plugin_dynamics_learning v2.0.0-rc.8

## v2.0.0-rc.8 — 2026-11-25

- 安全加固（Batch 1）：强制社交接口鉴权、补齐 `typing_habit` 防护、限制自动好感度变更并接入统一管理鉴权。
- 数据与存储修复（Batch 2）：精简 `learning` 事件载荷、补齐时间索引，修复清理与回滚逻辑。
- 性能与稳定性优化（Batch 3）：缩小乐观锁粒度、拆分 Provider 退避状态、新增只读钩子并清理死代码。
- 工程与发布治理（Batch 4）：增强 `theme.js` 空值安全、补齐离线 CI/打包门禁并同步系列共享资产。

## v2.0.0-rc.7 — 2026-11-24

- 根治熔断器风暴：区分瞬态 `ReviewDeferred(retry_after)` 与永久 `ReviewUnavailable`，`AutomaticReviews` 增加全局 `_budget_backoff_until` 与指数退避（60s->120s->300s）并跳过熔断期垃圾审计写入，批量审查遇熔断立即暂停保留队列。
- `learning.sqlite3` 启用 `PRAGMA auto_vacuum=INCREMENTAL`、`journal_size_limit=64MiB`、定期 `incremental_vacuum`/`wal_checkpoint(TRUNCATE)`，并新增终态 proposal 证据压缩/裁剪、`audit.superseded` 瘦身、`analysis_inbox` 与 `relation_source_receipts` 过期清理。
- 后台审查默认 `model_reasoning_effort='low'` 与分级 `max_tokens`（768/1536），移除 `system_prompt` 中的易变 `baseline_threshold` 以提升缓存命中；同步最新 `series_panels.py`。

## v2.0.0-rc.6 — 2026-11-22

- 接入全系列共享 `ProviderBudget` 并发池与遥测上报（`main.py` 经 `series_llm_call` 调用，队列繁忙时优雅延后为 `ReviewUnavailable`）。
- 下沉主控共享配置（SSOT）：手账与日统计窗口统一解析共享时区；模型解析遵循「显式配置 > 主控 `reply_provider`/`decision_provider` > 宿主默认」；观察、注入与总控矩阵全面联动 `exclude_groups` 排除护栏。
- 同步按语言拆包的 `series-messages.js` 轻加载器（`series-messages-{en,ja,ru}.js` 按需懒加载）及统一 `series_panels.py`。

## v2.0.0-rc.5 — 2026-11-21

- 合并线上 phase5 提示词缓存与低信号门控优化：跨群统一角色级 `system_prompt` 前缀、按 UTF-8 字节落实 22KB/24KB 载荷预算、跳过纯噪声低信号 ANALYZE 批次，并为模型调用附带确定性 2 消息静态缓存锚点。
- 升级群间系列 v2 面板联动，修复学习总开关关闭时群控矩阵状态与 `engine.allowed()` 不一致的问题：明确区分「已配置该群」与「当前实际会执行学习」，并在状态摘要与总览徽章如实反映总开关关闭态。
- 对齐世界手账与联邦查询的角色隔离及日期时区边界：统一以 `UTC+8` 解析自然日并支持显式 `[start_ts, end_ts)` 区间透传，严格过滤有效角色且禁止指定群未绑定角色时放宽为全角色查询。

## v2.0.0-rc.4 — 2026-11-20

- 优化分析与审查链路的前缀缓存与批量评审成本（DL-1..DL-3），并保留生产环境成本优化改进。
- 为 ANALYZE 与 REVIEW 引入精简的模型载荷投影，将统一的 `character_context` 上提至 `system_prompt`。
- 合并自动审查请求、限制默认分析轮数与单批审查上限（4 条 / 28KB），收紧输出篇幅并支持可选模型 Token/推理参数。

## v2.0.0-rc.3

支持在原版 AstrBot 4.29.4 的插件区域直接切换已更新的群间面板。入口和目标均需更新，左侧菜单保留入口。

同一会话内轮转待分析角色，优先处理当前角色。失败退避只影响该角色。

发布新的群间表达、习惯、喜恶、群文化与社交状态学习层，支持独立审核、工作台与回滚。相较公开的 1.x 影子调参插件为主要职责变化，升级前须备份旧插件、配置和数据；旧策略不会被自动解释为新版本的表达记忆。

这是发布候选。需要 AstrBot 4.29.4 或兼容的 4.x 版本。安装前保存旧插件目录、配置及对应 plugin_data；停止目标插件后安装，保留数据库与配置，再启用并检查加载日志。

如升级失败，先停用新版本并保留故障现场与新产生的数据；恢复升级前的代码与相匹配的数据备份。不要直接把旧数据库覆盖到运行中的插件，也不要假设较旧版本能够读取新版本写入的数据。

本候选的测试结果见发布批次验证报告。自动化检查不等于真实 QQ 收发或真实模型对话验收。

## 从公开 v1.4.1 升级

1.x 是使用 AstrBot 共享 KV 的影子调参插件，2.x 使用独立的 `learning.sqlite3` 保存表达与审查资料。请备份 AstrBot 的共享偏好数据库及插件配置；新版本不迁移旧 `learning_*_v1` 调参记录，也不应把它们当成新版本的词条或审批。旧 KV 保留，首次升级需重新检查会话和分析/审核模型配置。需要继续使用旧影子调参流程时，保留 v1.4.1。
