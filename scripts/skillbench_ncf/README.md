# SkillsBench NCF：锁定原料与无泄漏审计

这一目录中的第一阶段脚本只允许读取：

- `skills_1000` skill library；
- `skills_1000.tar.gz` 原始压缩包；
- `skillgraph_1000.json` 冷启动图。

它不会读取官方 87 个 SkillsBench 任务。

## 当前机器直接审计

```bash
python3 scripts/skillbench_ncf/build_source_manifest.py
```

## 换机器或本机运行数据丢失后恢复

```bash
bash scripts/skillbench_ncf/prepare_locked_sources.sh
```

该脚本从 `Eric068/SkillDAG` 的固定 revision 下载两个文件：

```text
46fbb5121d06ab5c6c3e1713f47ac62c8e57b75a
```

下载完成后必须通过 `configs/skillbench_ncf/source_lock.json` 中记录的
SHA256，随后才会解压和构建 manifest。上游即使修改 `main`，也不会静默换成另一批数据。

## 构建family-level train/dev划分

```bash
python3 scripts/skillbench_ncf/cluster_skill_families.py
```

该命令使用锁定的：

- 1000-skill manifest；
- 冷启动图；
- 官方 `text-embedding-3-large` embedding cache；
- `configs/skillbench_ncf/family_clustering.json`。

它不读取官方 SkillsBench 任务。family 只负责把近重复技能固定到同一 split，
不能被解释为正标签或可替代关系。

## 可提交的输出

```text
artifacts/skillbench_ncf/manifests/
├── skills_1000_manifest.jsonl
├── skillgraph_1000_manifest.json
├── source_audit_report.json
├── skill_family_manifest.jsonl
├── family_split.json
└── family_audit_report.json
```

大型 skill library、任务包、模型 checkpoint 和运行轨迹仍保存在 `.runtime/`，
不提交 Git。GitHub 保存的是恢复脚本、固定 revision、输入 hash、构建代码和审计结果。

## 隔离边界

路径规则位于：

```text
configs/skillbench_ncf/eval_isolation.json
```

所有后续的合成任务、Judge、embedding 和训练脚本都应复用
`source_policy.assert_training_source()`，不能自行绕过 allowlist。

## 构建零API Pilot

```bash
python3 scripts/skillbench_ncf/build_pilot_v1.py
```

完整数据写入Git忽略的：

```text
data/skillbench_ncf/pilot_v1/
```

可提交的抽样清单和审计报告写入：

```text
artifacts/skillbench_ncf/manifests/
├── pilot_v1_sample_manifest.json
└── pilot_v1_audit_report.json
```

该Pilot的所有task、query和pair均为 `training_ready=false`，不能直接训练。

## 构建零 API operation cards 与证据门控

```bash
python3 scripts/skillbench_ncf/build_evidence_v1.py
```

该步骤只读取锁定的 `skills_1000` 和 family split，不读取官方 SkillsBench
任务，也不调用任何 API。它会：

- 从 `SKILL.md` 的具体工作流、操作、API endpoint、tool slug 和示例中提取
  `operation_cards`；
- 按 A/B/C 对全部 1000 个 skill 做证据充足性分级；
- 仅在存在不同 operation card 时构造 train / in-family-dev 种子；
- 对证据不足的 skill 返回 B/C 类，不凭常识制造监督标签。

可提交的审计产物：

```text
artifacts/skillbench_ncf/manifests/
├── evidence_v1_audit_report.json
└── evidence_v1_review_sample.json
```

完整 operation cards 和任务种子属于可重建中间数据，写入 Git 忽略的：

```text
data/skillbench_ncf/evidence_v1/
```

## 审计 Train / In-family Dev 是否为独立操作

```bash
python3 scripts/skillbench_ncf/audit_operation_pairs_v1.py
```

该步骤仍然是零 API 预筛：

- 具有不同具体 service tool slug 的非顺序操作进入高置信集合；
- `Step 1/Step 2` 等疑似同一 workflow 阶段不会自动通过；
- 缺少独立工具证据的文本配对进入待审集合；
- 只有高置信 pair 才会输出到下一步 naturalization seeds。

完整 pair 审计保存在 Git 忽略的
`data/skillbench_ncf/operation_pair_audit_v1/`，小型报告和抽查样例保存在
`artifacts/skillbench_ncf/manifests/`。

## MiniMax-M2.7：10个配对的一次式自然化 Pilot

```bash
bash scripts/skillbench_ncf/run_naturalize_minimax_pilot_v1.sh
```

该命令固定使用云雾 profile 下的 `MiniMax-M2.7`，将10个 Train/Dev
高置信配对合并为一次请求，同时生成：

- 英文自然任务；
- 中文释义；
- grounded / independent 自检；
- 本地ID、泄漏和重复度检查。

这是为了节省费用的 `single_pass_self_audit`，不是独立第二模型 Judge。原始响应和
完整文本保存在 Git 忽略的 `data/skillbench_ncf/naturalization_v1/`，可提交报告
保存在 `artifacts/skillbench_ncf/manifests/`。

## MiniMax-M2.7：独立双 Judge

```bash
bash scripts/skillbench_ncf/run_judge_minimax_pilot_v1.sh
```

该命令使用两个彼此独立、且看不到生成器自评的 MiniMax 上下文：

1. Grounding Judge逐条核对能力证据、合成槽位和不支持声明；
2. Pair Judge核对Train/Dev是否对应不同操作、是否泄漏标识符。

它不再使用数值置信度。只有两个Judge与本地规则同时通过的pair才进入共识通过清单。

完成定向修订和复审后，可合并最终质量门控结果：

```bash
python3 scripts/skillbench_ncf/finalize_minimax_pilot10_v1.py
```

后续批次使用不含生成器自评分的V2生成器。例如第11–20组：

```bash
bash scripts/skillbench_ncf/run_naturalize_minimax_batch_v2.sh \
  --offset 10 \
  --limit 10
```

当前生成器实际使用 `v3_detailed_evidence` 协议：它保留 `SKILL.md`
代码块中的参数 schema，同时保持旧 operation ID 不变。生成器必须显式返回
`synthetic_slots`，不再返回 accept 或数字置信度；质量判断交给后续独立双 Judge、
本地规则和确定性人工审计。

第 11–20 组的完整可复现命令为：

```bash
bash scripts/skillbench_ncf/run_naturalize_minimax_batch_v2.sh \
  --offset 10 --limit 10 \
  --output-dir data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2 \
  --report artifacts/skillbench_ncf/manifests/minimax_m27_naturalization_batch02_evidence_v2_report.json

bash scripts/skillbench_ncf/run_judge_minimax_pilot_v1.sh \
  --items data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2/normalized_items.jsonl \
  --output-dir data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2/independent_judges \
  --report artifacts/skillbench_ncf/manifests/minimax_m27_batch02_evidence_v2_independent_judge_report.json

python3 scripts/skillbench_ncf/apply_minimax_pilot10_revisions_v1.py \
  --items data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2/normalized_items.jsonl \
  --config configs/skillbench_ncf/minimax_batch02_evidence_v2_revisions.json \
  --output data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2/revisions_v1/revised_items.jsonl

bash scripts/skillbench_ncf/run_judge_minimax_pilot_v1.sh \
  --items data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2/revisions_v1/revised_items.jsonl \
  --output-dir data/skillbench_ncf/naturalization_v1/minimax_m27_batch02_evidence_v2/revisions_v1/independent_judges \
  --report artifacts/skillbench_ncf/manifests/minimax_m27_batch02_evidence_v2_revision_v1_judge_report.json

python3 scripts/skillbench_ncf/finalize_minimax_batch02_evidence_v2.py
```

缓存存在且 prompt hash 一致时，生成和 Judge 命令不会再次调用 API。

## V3：增强 NCF 专用的 skill 表示

V3 不修改 SkillDAG 基线使用的原始 embedding，也不放宽正例 Top-20 门控。它仅为
NCF 构造第二个 skill 视角：本地通用 `SKILL.md` 描述加上与其明确链接的公开
Composio toolkit / Supported Tools 文档。公开 Suggested Prompts 不进入 skill
表示，官方 87 道 SkillsBench 任务仍保持隔离。

```bash
python3 scripts/skillbench_ncf/build_enriched_skill_texts_v3.py
bash scripts/skillbench_ncf/embed_enriched_skills_v3.sh
python3 scripts/skillbench_ncf/build_hybrid_skill_embeddings_v3.py
bash scripts/skillbench_ncf/build_enriched_pairs_v3.sh
bash scripts/skillbench_ncf/run_neumf_enriched_v3.sh
.envs/gos-ncf-eval/bin/python scripts/skillbench_ncf/evaluate_full_catalog_v3.py
```

embedding 命令会把 574 段公开文档派生文本发送到所配置的外部 API，因此必须在
获得数据发送授权后运行；其余命令不调用 API。混合表示固定为原始向量与增强向量
各 0.5，目的是保留服务身份锚点，而不是通过放宽门控恢复覆盖率。

`full_catalog_v3_comparison.json` 是必要的失败保护：训练时的五候选 dev 指标不能
代表在完整 skill library 上的排序能力。当前 V3 证明表示增强改善了 full-catalog
cosine，但现有四负例 NeuMF 在全目录排序上失效，后续必须改善训练负例覆盖，不能
把五选一指标写成正式收益。

## V4：与 ALFWorld 对齐的多源候选池

```bash
.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/build_candidate_pool_v4.py
.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/check_candidate_pool_v4.py
```

该步骤与 ALFWorld V3 一样合并两种 context cosine、冻结旧 NCF 的 hard
candidate，以及 `depends_on/composes_with` 一跳图邻居。SkillsBench 额外加入4个
确定性 easy probe，并始终限制在相同 family split。所有输出均为
`candidate_only=true, label=null`，source skill 只作为派生数据的 provenance anchor，
不能冒充已经完成的 Judge 标签。

当前原始池平均每个 query 约30个候选并覆盖全部1000个skill，但冻结 V2 NCF
存在明显的全局热门偏置。因此报告会保持 `judge_ready=false`；在做付费 Judge 前，
必须先完成模板去重或候选曝光控制，不能直接把约11万条 pair 全部送给 API。

曝光均衡后的候选池及其30-query固定试标清单使用：

```bash
.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/build_balanced_candidate_pool_v4.py
.envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/build_graded_judge_pilot_v4.py
.envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/check_graded_judge_pilot_v4.py
```

固定清单按 query 来源、train/dev family split、source 是否需要锚点以及图候选
分层；选择由 query ID 的稳定 hash 决定，不按模型结果挑题。每个 query 最多保留
20个候选，且在发给 Judge 的 payload 中隐藏 source skill provenance。

获得付费调用授权后，使用 MiniMax-M2.7 做一次 2/1/0/uncertain 小样本质检：

```bash
bash scripts/skillbench_ncf/run_graded_judge_pilot_v4.sh
.envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/audit_graded_judge_pilot_v4.py
```

该命令一条 query 一次请求，不使用数值置信度。原始响应与标签保存在 Git 忽略的
`data/skillbench_ncf/graded_judge_pilot_v4/`；可提交报告保存在
`artifacts/skillbench_ncf/manifests/`。Pilot标签在完成确定性校准和审计前保持
`training_ready=false`，不能直接并入正式训练。

审计不会把派生 query 的 source skill 自动升级为2。如果 Judge 判断 source
能力证据不足，该 query 应退出或重新生成。冻结旧NCF分支只负责提出 hard
candidate，不提供正标签；数值置信度不采集、不进入模型。

## V4完整3,650-query分级Judge

在Pilot通过、且用户明确授权完整付费调用后：

```bash
.envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/build_graded_judge_full_v4.py

bash scripts/skillbench_ncf/run_graded_judge_pilot_v4.sh \
  --manifest data/skillbench_ncf/graded_judge_full_v4/judge_manifest.jsonl \
  --output-dir data/skillbench_ncf/graded_judge_full_v4/judgments \
  --report artifacts/skillbench_ncf/manifests/graded_judge_full_v4_results.json \
  --limit 3650 \
  --workers 128
```

并发只减少等待时间，每条query的20候选证据、Prompt、temperature=0和分级协议
完全不变。响应按query hash独立缓存；中断重启不会重复请求已经返回的query。
若模型遗漏候选、返回清单外ID或重复ID，对应未获得有效唯一判断的候选一律记为
`uncertain`，不猜测0/1/2，也不参与训练。
128路并发用于约20分钟墙钟时间目标；如果服务端返回限流，失败query保留为未完成，
随后仅对这些query降低并发补跑，不把HTTP错误当成标签。
首轮结构失败使用完全相同的Prompt单独重试，随后再合并为完整标签：

```bash
.envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/build_graded_judge_retry_v4.py

bash scripts/skillbench_ncf/run_graded_judge_pilot_v4.sh \
  --manifest data/skillbench_ncf/graded_judge_full_v4/retry/judge_manifest.jsonl \
  --output-dir data/skillbench_ncf/graded_judge_full_v4/retry/judgments \
  --report artifacts/skillbench_ncf/manifests/graded_judge_full_v4_retry_results.json \
  --limit 47 --workers 48

.envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/merge_graded_judge_full_v4.py
```

## V4：全量 Judge 标签校准、加权与 NeuMF 数组

全量 MiniMax-M2.7 Judge 完成后，使用一个零 API、fail-closed 的入口：

```bash
bash scripts/skillbench_ncf/prepare_neumf_full_v4.sh
.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/preflight_neumf_full_v4.py
```

它依次执行：

1. 保留 `base_cosine`、`hybrid_cosine`、冻结旧 NCF 三个运行时候选分支；
2. 排除 `uncertain`、纯图邻居、纯 source anchor、纯 easy probe；
3. 不把 query 的 source skill 自动强制为 2；
4. 排除没有保留 grade-2 skill 的整个 query group；
5. 检查 train/dev family 隔离和规范化 query 文本重叠；
6. 只用 train 统计计算
   `1/query_pair_count × query-source逆频率 × skill逆平方根频率`；
7. 在 train 权重 P99 截断并重新归一化，防止稀有来源支配训练；
8. 组装内容型 NeuMF 使用的 query embedding 与增强 skill embedding。

大型 pair、embedding 和 `arrays.npz` 均保存在 Git 忽略的 `data/` 下。Git 只
提交脚本和不包含任务文本的聚合审计报告。官方 87 个 SkillsBench 任务仍未被读取。

预检通过后，正式训练入口为：

```bash
bash scripts/skillbench_ncf/run_neumf_full_v4.sh
```

该入口直接复用 ALFWorld 81.43% V3 的加权 pointwise BCE、query 内
`2>1>0` pairwise ranking、GMF/MLP 预训练和 NeuMF 融合训练流程。训练完成后
还会只用 train query 计算全部1000个skill的通用 logit prior、用内部dev选择
cosine/NCF融合系数，并导出容器可直接读取的 portable bundle。

## 构造候选对与全技能自监督覆盖

自然任务候选池需要 embedding API；存在相同缓存时不会重复计费：

```bash
bash scripts/skillbench_ncf/build_candidate_pairs_v1.sh
```

随后可用零 API 的文档自监督补齐全部 1000 个 skill：

```bash
python3 scripts/skillbench_ncf/build_self_supervised_queries_v1.py
```

该步骤直接从每个锁定的 `SKILL.md` 构造能力 query，并执行以下约束：

- 全部 1000 个 skill 至少有一个正例视图；
- A/B/C 三类证据权重分别为 `0.35/0.08/0.20`，不能等同自然任务 gold；
- B 类仅学习服务域和工具发现，不臆造具体业务操作；
- 过短的操作片段不会生成第二视图；
- family-dev 数据在模型选择期间保持隔离，超参数冻结后才能并入最终训练；
- 不读取官方 87 道 SkillsBench 评测题。

完整 query 写入 Git 忽略的 `data/skillbench_ncf/self_supervised_queries_v1/`；
可提交的统计报告和抽查样本写入 `artifacts/skillbench_ncf/manifests/`。

计算 query embedding、构造保守 hard negatives 并与自然任务正例合并：

```bash
bash scripts/skillbench_ncf/build_self_supervised_pairs_v1.sh
```

默认每个合格的自监督 query 使用 1 个正例和 4 个 hard negatives。负例必须：

- 与正例处于相同的 family train/dev split；
- 来自不同 family；
- 与正例不存在直接图边；
- cosine 分数低于已知正例。

正例在本 split 中排到 Top-20 之外的视图会退出训练，但只要同一 skill 的基础
capability 视图仍然合格，就不会损失该 skill 的监督覆盖。90 个自然任务在这一阶段
只合并已经确认的 source 正例；`needs_judge` 候选不会被擅自当成负例。
embedding 与完整 pair 数据保存在 Git 忽略的
`data/skillbench_ncf/self_supervised_pairs_v1/`。

## 训练小规模 content-NeuMF Pilot

```bash
bash scripts/skillbench_ncf/run_neumf_pilot_v1.sh
```

该命令不调用 API，依次完成：

1. 从已有 embedding 缓存生成 NeuMF 数组；
2. 仅使用 train split 更新梯度；
3. 使用 held-out family dev 的自监督候选组早停；
4. 对比 cosine、task-agnostic skill popularity 和 NeuMF。

仅有正例而没有 Judge 候选集合的 45 条 natural-dev 数据不会进入内部排名指标，
否则会形成单候选必然满分。checkpoint 保存在 Git 忽略的
`.runtime/skillbench_ncf/models/neumf_pilot_v1/`，公开模型数据报告与训练摘要保存在
`artifacts/skillbench_ncf/manifests/`。

这个 Pilot 只能检查模型是否学到 held-out-family 的 task–skill 交互，不能替代
官方 87 题的 Agent 执行评测。由于 hard negatives 被保守约束为 cosine 低于正例，
该内部 dev 上的 cosine 具有结构性优势，不能据此比较论文主表性能。

## 扩充 linked-public query 并训练 V2

通用 Rube skill 可以沿锁定 `SKILL.md` 中的显式链接，缓存公开 Composio
toolkit Markdown：

```bash
PYTHONPATH=scripts/skillbench_ncf \
  .envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/fetch_composio_toolkit_evidence_v1.py

PYTHONPATH=scripts/skillbench_ncf \
  .envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/build_expanded_queries_v2.py
```

该变体必须在论文中标为 `linked-public evidence`，不能写成严格的
`local SKILL.md only`。公开 prompt 默认权重为 `0.10`；权重 `0.25` 的实验会明显
改善扩充 dev，却损害原本 local dev，说明外部任务分布会覆盖旧能力。

V1/V2模型的固定交叉评测命令为：

```bash
PYTHONPATH=src/GraphOfSkills_NCF \
  .envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/compare_neumf_v1_v2.py
```

全部公开文档原文、query、embedding、pair、数组和checkpoint仍保存在Git忽略的
`data/`或`.runtime/`；Git只保存下载/解析代码、URL与SHA256统计和小型实验报告。

## 接入原始 SkillDAG SkillsBench 评测

前置阶段规划、portable NeuMF、每次 search 重排和原始 Harbor 评测链路的实现与命令见：

```text
docs/skillbench_ncf_native_skilldag_integration.md
```

该接入不新写 Verifier，也不替换原始 `run_skillsbench.sh → Harbor` 主链路。

## 汇总已通过质量门控的自然任务

第 21–46 个完整 train/dev pair 使用确定性修订和选择性双 Judge 封版：

```bash
python3 scripts/skillbench_ncf/finalize_minimax_batch03_evidence_v2.py
python3 scripts/skillbench_ncf/finalize_minimax_batch04_evidence_v2.py
python3 scripts/skillbench_ncf/finalize_minimax_batch05_evidence_v2.py
python3 scripts/skillbench_ncf/build_naturalized_corpus_v1.py
```

完整任务写入被 Git 忽略的 `data/skillbench_ncf/naturalized_corpus_v1/`；Git 只保存
构建代码、修订配置、计数和输入/输出 SHA256。当前语料为45个通过质量门控的skill、
90个自然任务，并且没有读取官方 SkillsBench 87题。

## 构造 V5 typed-workflow 排序候选

以下阶段不调用 API，也不读取官方 87 道评测题：

```bash
python3 scripts/skillbench_ncf/build_typed_operation_inventory_v5.py
python3 scripts/skillbench_ncf/mine_typed_workflow_candidates_v5.py
python3 scripts/skillbench_ncf/tier_typed_workflow_candidates_v5.py
python3 scripts/skillbench_ncf/assemble_typed_ranking_candidates_v5.py
python3 scripts/skillbench_ncf/audit_typed_ranking_candidates_v5.py
python3 scripts/skillbench_ncf/assemble_typed_atomic_ranking_candidates_v5.py
python3 scripts/skillbench_ncf/audit_typed_catalog_coverage_v5.py
```

当前 workflow 层生成 477 个工作流、1,908 个 query 和 11,448 个六候选排序
pair；atomic 层再提供 673 个 query 和 4,038 个 pair，将 V5 高证据 skill 覆盖从
219 提高到 673。旧的低权重自监督仍保证全部 1000 个 skill 可见。完整数据位于
Git 忽略的 `data/skillbench_ncf/typed_*_v5/`，可提交 manifest 位于
`artifacts/skillbench_ncf/manifests/`。这些记录全部是 candidate-only，必须在
query embedding、实际 rank-pressure 筛选和标签校准完成前保持
`training_ready=false`。

获得对合计 2,581 条新 query 的 embedding 调用授权后，再运行：

```bash
python3 scripts/skillbench_ncf/embed_typed_ranking_queries_v5.py
python3 scripts/skillbench_ncf/screen_typed_rank_pressure_v5.py
python3 scripts/skillbench_ncf/screen_typed_atomic_rank_pressure_v5.py
```

筛选只保留困难对照超过（或距离不超过 0.02）必需 skill，以及必需 skill 掉出
cosine Top-20 的排序单元。筛选不会自动确认标签，输出仍保持
`training_ready=false`。

## V5 public-prompt 全量 Judge、校准与训练数组

全量多 operation 证据 Judge 包含 1,053 个 query、9,169 个候选 pair。付费响应
缓存在 `data/skillbench_ncf/public_prompt_multievidence_full_v5/minimax_m27/responses/`；
再次运行 Judge 会按 prompt hash 命中缓存。Judge 结果必须先保持
`training_ready=false`，再执行以下零 API 步骤：

```bash
PYTHONPATH=scripts/skillbench_ncf \
  .envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/calibrate_public_prompt_multievidence_v5.py

PYTHONPATH=scripts/skillbench_ncf \
  .envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/prepare_neumf_public_prompt_v5.py

PYTHONPATH=scripts/skillbench_ncf \
  .envs/skilldag-debug/bin/python \
  scripts/skillbench_ncf/check_no_eval_leakage_public_prompt_v5.py

.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/preflight_neumf_public_prompt_v5.py
```

固定校准规则为：不强制 source 为 grade 2；排除 `uncertain`；正例必须保留
结构化 Judge 证据；每个 query 必须同时具有 grade 2 和更低等级以形成严格
preference。最终得到 868 个 query、7,537 个 pair，其中 train 为 708/6,290，
dev 为 160/1,247。训练权重只使用 train 统计量：

```text
1 / query候选数 × grade factor(0:1, 1:2, 2:3)
× inverse-sqrt(skill exposure)
→ train P99截断
→ train均值归一化为1
```

该数据是 V5-only 主训练输入，不与 V4.1 的 51,911 对默认混合。内部 dev 仅用于
checkpoint 选择；官方 87 题保持未读，必须等模型和融合参数冻结后再评测。

正式训练继续复用锁定的 ALFWorld V3 pointwise + pairwise NeuMF 实现：

```bash
.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/train_neumf_full_v4.py \
  --arrays data/skillbench_ncf/model_data/neumf_public_prompt_v5/arrays.npz \
  --output .runtime/skillbench_ncf/models/neumf_public_prompt_v5 \
  --public-report artifacts/skillbench_ncf/manifests/neumf_public_prompt_v5_training_report.json \
  --report-schema-version skillbench_ncf.neumf_public_prompt_training.v5 \
  --experiment-label 'SkillsBench V5 public-prompt'

.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/calibrate_neumf_full_v4.py \
  --arrays data/skillbench_ncf/model_data/neumf_public_prompt_v5/arrays.npz \
  --model-dir .runtime/skillbench_ncf/models/neumf_public_prompt_v5 \
  --public-report artifacts/skillbench_ncf/manifests/neumf_public_prompt_v5_calibration_report.json \
  --report-schema-version skillbench_ncf.neumf_public_prompt_online_calibration.v5 \
  --prior-mode none \
  --alpha-grid 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 \
               0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00

.envs/gos-ncf-eval/bin/python \
  scripts/skillbench_ncf/compare_public_prompt_v5_fusion.py
```

V5 的 train popularity 审计较弱（Top-1/Top-10 grade-2 占比为 1.03%/8.98%），
因此不沿用 V4 的 skill-prior 扣除；使用原始 NeuMF logit 与 cosine 做 query 内
标准化融合。内部 dev 选择 `alpha_cosine=0.55`，NDCG@3 从 cosine 的 0.5780
提高到 0.6528。该结果与配对 bootstrap 都是 alpha 选择后的内部诊断，不能写成
官方 SkillsBench 最终测试结论。
