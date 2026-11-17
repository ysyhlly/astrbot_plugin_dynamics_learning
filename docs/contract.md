# Dynamics Learning contract v1

Chat Dynamics 通过 AstrBot 当前插件注册表查找 `astrbot_plugin_dynamics_learning` 的激活实例，调用公开能力方法，不导入该插件模块、读取其数据库或写入其配置。

| 方法 | 约定 |
| --- | --- |
| `learning_capabilities()` | 同步返回 schema_version=1、ready 与能力列表 |
| `ingest_dynamics_event(event)` | 同步校验并入有界队列，返回是否接受；不调用模型、不访问磁盘 |
| `get_learning_context(scope, persona, member, baseline, host_version)` | 同步读取内存快照，返回 context、可选 threshold 和所用提案版本 |

事件包括 `observed`、`decision`、`outcome`。字段包含作用域、人格指纹、发言者、回合/消息 ID、主插件版本与配置摘要、实际后端、是否强制回应、有界原文、结构化决策、真实发送结果与关联 ID。结果记录追加，不覆盖原始判断。实际发送文本只在成功回执之后发布。未知信息不填成否定或零。

当前实现覆盖 Chat Dynamics 的人格群聊和私聊主链；旧 legacy 回复路径不应用学习内容。一次回合可包含多条 outcome，观察计数按 turn_id 去重；已经成功发送的回合不会因后续失败被改成从未发送。

回调绑定原伙伴实例，重载后旧回调不可写入新实例。学习读取在回合开始固定；后续审批不会悄悄改变本轮输入。有学习增量的回合在生成/发送校验处核对当前实例及版本。可选插件缺失、未就绪或异常时，基础聊天保留。

模型分析采用严格 JSON 的 proposals 列表，最多五项。候选类别为 style/jargon/relationship/state/persona/decision_guidance/participation。字段引用的 evidence_ids 必须存在于本批输入，不能跨 scope/persona；成员提案只使用该成员发言证据。模型审查必须返回 verdict、reason、risks，verdict 仅允许 approve/reject/defer。

审查流程：pending → reviewed → applied；拒绝进入 rejected；人工修改新建 pending 并拒绝旧提案；应用后的版本可以 rolled_back。LLM 模式允许 reviewed 后自动应用，其他模式要求管理员批准。状态转换校验 updated 时间与学习项 revision，防止延迟审查和并发操作覆盖新状态。

后台路由位于 `/api/plug/astrbot_plugin_dynamics_learning/`：GET overview，POST analyze/review/action/amend/purge。所有路由都要求当前 Dashboard 管理员会话。`action` 需要 id、action 和页面读取到的 updated，支持 approve/reject/rollback；没有人工逐条标注接口。
