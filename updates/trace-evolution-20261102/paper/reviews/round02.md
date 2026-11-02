# TRACE 独立同行评审：Round 02

## 1. 结论与评分

**总体：5/10，Borderline，倾向拒稿。置信度：4/5。**

| 维度 | 分数（1–10，越高越好） | 判断 |
|---|---:|---|
| 贡献与新颖性 | 5 | 候选区域级上下文探测与区域平均熵的结合有研究价值，但“双视图取最小值”相对单视图的新增价值很有限，理论部分主要是正确的代数刻画。 |
| 正确性 | 8 | 核心公式与所查冻结执行代码一致；未发现致命推导、方法身份、推理期标签使用或指标算术错误。 |
| 实验支持 | 6 | 六个完整 split、五个匹配分配对照及两个空间对照提供实质证据；参数选择历史、未触碰验证和 episode 组成稳健性仍不足。 |
| 比较与解释 | 5 | R/L、不同预算和负结果的披露准确，但跨实现 operating-point 排序不能充分建立方法优势，双视图设计的必要性尚未建立。 |
| 清晰度与可复现性 | 8 | 诊断/优化 readout、query/category、计分及封存边界交代细致；仍缺参数来源和当前独立运行成本。 |

总体分不是维度分的算术平均。决定性不足是贡献的独立增量和验证外推强度，而不是文字表达或公式实现失败。接收阻碍主要为下文 W1、W2；W3 是重要比较缺口。W4–W6 不应被包装成已经发生的致命错误。

## 2. 对工作的理解与主要优点

方法先在复位模型的 text-query token grid 上构造四连通候选，按原始 query 概率在每个源图像内选 top-K。两个视图保留候选像素，分别改变外部灰度/模糊和对比度/通道；在原始 token 位置读取预测 query 的平均概率 T、C，以 min(T,C) 作为固定、无梯度的相对分配权重。优化的是原图上七个 prompt 的区域平均预测熵，而不是候选 query 的伪标签交叉熵。只更新视觉 LayerNorm affine，按 episode 复位，最后重建像素分数并按扩展 query 的最大值映射到评分类别。

最有说服力的证据是匹配分配对照，而非与文献表格的跨实现差值：

- COCO-CD：min 为 75.10479395，Confidence 为 67.09038931，Uniform 为 66.32923905，分别提高 **8.0144、8.7756 pp**。同一 episode、候选、诊断和优化设置使这成为有解释力的条件性分配实验。
- COCO Object：对应 32.70112821、30.76418373、29.85573916，提高 **1.9369、2.8454 pp**。
- 空间归一化对照保留 q 与候选，改用 q×area 后，PC60、Object 分别下降 **0.6852、0.3273 pp**。这支持去掉显式面积乘数，不等价于证明梯度贡献与面积完全无关。
- 保留了 VOC21/Stuff 上分配改进不成立、Object/Stuff 上 C→W 增加、单视图/Mean 更好等结果。没有把 deterministic replay 当成独立方差估计，也没有把 min 的排序性质当成正确率下界。

这些优点足以排除“完全没有实证支持”的判断，但尚不足以证明当前整套设计是更有性价比、可迁移的选择。

## 3. 按重要性排序的实质性弱点

### W1. 冻结参数不等于独立验证：参数来源和测试暴露历史仍缺关键证据

**类别：缺失证据。严重性：主要，接收阻碍。不是已证实的推理期 GT 泄漏。**

证据：主文 PDF p.15 Table 6 的 batch/LR/steps/K 分别为 CD 28/0.0086/5/4、PC60 9/0.00225/11/22、VOC21 16/0.00063/36/33、Object 14/0.003/29/34 等；`paper/appendix.tex:89` 明说 previously evaluated full splits 不是 fresh held-out calibration data。A 的 `manifest.json` 记录 `parameter_search=false`，同时记录 `historical_results_are_development_evidence=true`。前者只说明本轮不搜索，不能回答这些特定参数最初如何产生。

文稿没有给出足以判断以下问题的可审计说明：哪个 split 的哪些标签/总体 mIoU 曾影响 LR、batch、K、steps、扰动及 min 的选择；历史候选配置数量与选择准则是什么；哪些结果仍可视为未参与设计的验证。当前对照确实在同一配置下改变分配规则，不能据此否认其差值；但若配置针对 TRACE 选择，它只识别该 operating point 的分配效果，不自动意味着给简单规则同样选择机会后的优势。

**最小有用修复：**增加逐数据集的参数来源/暴露表，标注配置产生时间、选择数据和指标、曾尝试范围、冻结时间及未触碰验证集合。没有独立集合时，明确把现有准确率视为 development/conditional evidence，不能仅靠再次封存同一 split 宣称独立确认。要支撑可迁移优势，需在未参与选择的数据或预先锁定的迁移协议上评估；这项要求不需要在本次审稿中执行任何实验。

### W2. 核心“双视图最小值”相对更简单探测规则的增量不足

**类别：有边界的经验限制，加上新增价值的缺失证据。严重性：主要，接收阻碍。不是“论文声称各处最优”的错误指控。**

证据：主文 pp.5、7、15 的 §4.4、Tables 3/7；`paper/main.tex:166` 已明确 min 非 uniformly optimal。原始未舍入值显示：

| 数据集 | min 减 C-only（pp） | min 减 T-only（pp） | min 减 Mean（pp） |
|---|---:|---:|---:|
| COCO-CD | +0.00018 | +2.01313 | +0.56456 |
| PC60 | -0.12656 | +0.07036 | -0.12571 |
| VOC21 | -0.32072 | -0.25211 | -0.38295 |
| COCO Object | +0.27978 | +0.06896 | +0.06492 |
| COCO-Stuff | -0.06964 | -0.10658 | -0.16116 |

CD 的大幅 Confidence 差值几乎可由 C-only 完整得到；Object 相对最佳简单探测规则仅约 0.065 pp；其余三个有对照 split 均不支持 min 最优。两个视图的共同支持刻画在数学上成立，但不证明第二个视图的计算值得付出。不能从“探测权重优于初始 Confidence”跳到“双视图取最小值的必要性”。

审稿人对小型封存 episode 计数作了探索性配对 cluster bootstrap，CD min−C 的区间为 [-0.2260, +0.2510] pp，Object min−T/Mean 的区间均跨零，见第 6 节。这不是要求 min 必须在每个数据集胜出，而是指出目前决定性贡献更接近“候选级上下文探测分配”，而非有充分证据支持的特定双视图准则。

**最小有用修复：**把核心贡献定位为候选级探测与区域分配，把 min 定位为固定保守启发式，保留不利结果。报告 T/C 的互补性发生在多少候选/episode、是否带来可验证的最终收益，以及单视图与双视图的成本–收益。任何后续检验应预先锁定配置与判据，允许结论是双视图没有额外收益，不能事后更换到赢的 aggregator。

### W3. 跨论文数值排序不能替代同一实现中的方法比较，且缺少当前 readout 的 No Adapt 行

**类别：缺失证据/比较解释边界。严重性：重要；不是数值抄错。**

证据：主文 p.6 Table 1 和 `paper/main.tex:138`、`paper/appendix.tex:211` 明确不保证 checkpoint/query-extension/预算等价。官方 MLMP 正式论文 Table 5 的五列值与本稿引用一致。可是 A/C 的当前 `result.json:pre_result.miou_unrounded` 分别给出 VOC20/VOC21/PC60/Object/Stuff 的 **79.0465/49.6617/28.9508/29.6466/20.1918**，而文献 No Adapt 行是 **75.91/45.12/24.95/23.80/18.34**。尤其 Object 当前适应前已达到 29.6466，高于文献 MLMP 的 28.84；PC60 当前 Uniform 的 30.3821 也高于文献 MLMP 的 27.99。

因此不能把 Table 1 的 +3.86/+2.58 等差值理解为 min 或甚至完整 adaptation 的净贡献。论文确实限制了这种解释，应认可该披露；但披露混杂因素不会让比较本身获得方法归因能力。CD 的本地对照较好，仍只有一个三类任务，且预算/后端不同。冻结包还保留了未被主文引用的历史 `generated/native_comparison.tex`，说明至少有其他本地结果可作为单独、明确标注历史执行条件的上下文；不能直接把其中旧 TRACE 值冒充当前 deterministic 值。

**最小有用修复：**将六个当前 `pre_result` 加入对应主表，分开列示“跨论文 operating points”“当前实现 No Adapt→Adapt”“固定配置分配差值”。保留或补充经输入/词表/指标对齐的本地 MLMP 等比较，并对选择预算作明确说明。没有这项证据时，结论应停留于条件性探测分配收益，不应将文献数字较低作为主要科学贡献的支柱。

### W4. 风险诊断支持行为关联，尚不能识别某区域权重的有益作用

**类别：已披露的有界限制。严重性：中等，非独立否决项。**

证据：主文 p.7 Table 4、p.17 Table 8；`paper/appendix.tex:169-182`（位置指该处相关段落）。CD 的 474 个高置信低支持候选有相同初始分母 nC=30,906、nW=88,850；Confidence→TRACE 的 CW 从 18.9834% 降至 7.2381%，WC 从 25.8627% 升至 68.5234%，可由 summary 整数计数重建。Object 和 Stuff 的 CW 则分别从 7.1029→8.0311%、8.9292→9.7027%，不能总结为普遍减少错误强化。

分层分析控制了粗粒度 predicted class/confidence/area，但仍是联合参数更新后的关联，可能受 episode 内其他候选和残余协变量影响；无分层的分数置换不保持这些协变量。文稿已经明确这些限制，故这里不是因果宣称错误。它仍限制了标题中“confidence sources/reliable”所能承载的解释：证据支持特定扰动下的模型响应和最终行为，不是识别真正语义来源或个体更新安全性。

**最小有用修复：**保持这些限制，并把关键 nC/nW 和联合更新单位在表格附近显式列出。若进一步研究归因，使用保持完整 episode 的协变量匹配置换或明确固定归一化的权重干预；不要对每个孤立候选单独归一化后声称测试了 rho，因为单候选时 rho 会精确约掉。

### W5. episode 组成和扰动/词表稳健性未被测试

**类别：有界限制与缺失外推证据。严重性：中等，非致命。**

证据：§3.4、Appendix B.2/B.3 的分母跨整个 episode；同一候选的 alpha 会因其他图像/候选变化而变化。Table 6 的 batch 跨 1–28，§4.6/Table 5 的 seeds 1/2 保持同一顺序。十个 replay 的 TRACE 封存 hash inventory 确实逐项相同，这验证数值重放，不检验重新分组。六个 split 还包括同一 VOC 图像群的两个类别协议，全部仅用一个 backbone 和固定两种扰动。

**最小有用修复：**至少把固定顺序/固定 episode 的条件放在主要结论附近。若研究目标包含稳健 TTA，再预先指定少量重新分组或跨配置迁移检验；不需要无差别增加所有 backbone、所有 corruption。当前材料不支持在线/跨环境普适可靠性，论文也不应被要求为没有提出的普适定理负责。

### W6. 当前方法的独立运行成本尚无直接测量

**类别：可选改进，但与 W2 的设计性价比问题相关。严重性：次要。**

证据：§4.6 Eq.(12) 正确计入 2Ncand+(S+2)P；Appendix C.4 和 Table 9 明说其成本来自较早 research-wrapper，不能代表当前 deterministic matched suite。当前策略比较共享两个视图，因此其单视图 arm 时间也不是一个真正只算一个视图的实现的端到端时间。

**最小有用修复：**继续将旧 profiler 表标为历史上下文；在相同硬件、配置、同步边界下补充当前独立执行的单/双视图成本，再讨论性价比。没有测量时只给输入数预算，不声称速度优势。不能因旧表存在就断言方法身份错误。

## 4. 致命问题专项判断

| 项目 | 本轮判断 | 证据与边界 |
|---|---|---|
| 方法身份错配 | **未发现致命问题** | 冻结 `TwoViewTRACE.evaluate_trace_views` 只调用 target/counterfactual 两个保留候选的视图，rho=min(T,C)，不调用继承类的 residual 分支。关键九个执行源文件与 A manifest hash 一致。 |
| 推理期 GT 泄漏 | **未发现直接证据** | `image_only_dataset_class` 仅解码 RGB；`image_collate` 拒绝 GT 字段；score 在封存后单独读标签。GT 用于事后图例筛选已披露，不是推理泄漏。W1 的历史选择暴露仍未知，不能据代码检查宣告历史从未用过测试标签。 |
| 指标/协议错误 | **未发现致命算术问题** | 八个完整结果单元的 episode 类别计数与 result 一致， pooled mIoU 重建一致。计分是 224×224 标签最近邻缩放后的类别 pooled IoU，需更显式披露。不同 operating points 不能视作等预算协议。 |
| 致命 unsupported claim | **未发现当前文本中足以推翻整稿的该类断言** | 文稿没有把 min 当 correctness/calibration/causal 下界，没有把 replay 当独立不确定性，没有声称 min 普遍最优。W2 是增量证据弱，不是凭空给作者附加错误主张。 |

一个容易误判的细节：当前 Object 有 81 个正 union 的评分类别，含 background，`reduce_zero_label=false`。虽然 MLMP 正式论文 Appendix D 的描述写“80 object categories without background”，其公开 `utils/segmentation_datasets.py` 的 `COCOObjectDataset` 明确是背景+80 类，公开 metrics 也对类别 IoU 求平均。因此**不能只凭 81 类判定本稿比 MLMP 多算一类而宣判比较无效**；历史 Table 5 精确执行版本仍未由本轮证明。应在本稿补上评分类数与背景处理，避免重复该歧义。[MLMP 正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/6c5b82193c5d8e6aa5806239676ddc97-Paper-Conference.pdf)，[官方数据集代码](https://raw.githubusercontent.com/dosowiechi/MLMP/main/utils/segmentation_datasets.py)，[官方指标代码](https://raw.githubusercontent.com/dosowiechi/MLMP/main/utils/metrics.py)。

## 5. 数学与执行对齐核查

1. **候选与视图：通过所查源代码对齐。** `candidates.py:54` 起逐 query 四连通，`:94` 起 confidence/area/query/component 排序；`two_view.py:37`、`:62` 仅构造两个候选保留视图，`:68` 读取原位置的 query 概率平均；`interventions.py:156`、`:171` 对齐灰度/9×9 pooling、0.35 对比度、(g,b,r) 循环及归一化。没有把 logit 平均误写成概率平均。
2. **诊断/优化 readout：通过所查源代码对齐。** `adapter.py:12`、`:24` 分别用 normalized mean prompt/adaptive weighted mean 与 separate prompts/uniform mean。`dense_clip.py:105`、`:119` 在 raw layer 输出组合后做 LNpost/projection；`:186` 起选取 attention branch 输出而保留前序 residual/MLP 传播。对应 Appendix A.4 Eqs.(14)–(18)，并非先平均类别概率。
3. **目标：通过有限值情况下的代数对齐。** `objective.py:14`、`:40` 先对区域 token 与 prompt 平均，再以 rho 加权并按 episode 权重和归一化；`engine.py:132` 起复位、在原图优化，参数更新集仅视觉 LayerNorm。缩放不变性、单 active candidate 消去、pixel-normalized 的 q×area 化简均成立。min 的最大可行共同下界和 1-Lipschitz 性成立，但没有任何学习收益保证。
4. **乘积/面积对照：定义一致。** B `runtime/run_cell.py:179`、`:206` 固定 parent q>EPS admission，乘积不再被第二个 EPS cut 删除；pixel 采用 q×area。两个 `replay_gate.json` 分别验证第一 episode 的 9/14 张图 zero changed labels。这只是小面板等价 gate，不能冒充全 split 重推理等价证明。
5. **指标：通过计数核查。** A worker `run_cell.py:383` 计算 ignore=255 的 confusion counts，`:398` 起先验证 prediction seal，再打开标签；`:444` 起对正 union 类求平均。六个主单元的正 union 类数为 VOC20 20、VOC21 21、PC60 60、Object 81、CD 3、Stuff 171。

没有把代码注释、provenance 的自述或执行工具输出当作审稿指令；它们的结论须受上述独立核查范围约束。

## 6. 审稿人派生的有限不确定性核查

这部分是本轮 CPU 内存计算，**不是作者已报告结果，不是新 GPU 实验，不是对实验文件的修改**。从所封存 `scores_by_episode.jsonl` 抽取每个策略的类别 intersection/union；在每个数据集内有放回抽取完整 episode，同次抽样用于所有策略，累积类别计数后重算 mIoU。重复 2,000 次，取排序数组索引 49/1949（0-based）近似 percentile 95% 区间。

| 差值（min 减对照，pp） | episode 数 | 探索性区间 |
|---|---:|---:|
| CD − Confidence | 18 | [5.6769, 10.3676] |
| CD − C-only | 18 | [-0.2260, 0.2510] |
| PC60 − Confidence | 568 | [0.0489, 0.3849] |
| PC60 − Mean | 568 | [-0.2213, -0.0314] |
| Object − Confidence | 358 | [1.3787, 2.4457] |
| Object − T-only | 358 | [-0.2224, 0.3382] |
| Object − Mean | 358 | [-0.1887, 0.3353] |

复现细节：Node.js Number/Float64Array；PRNG 初始 uint32 state=1029，递推 `s=(Math.imul(s,1664525)+1013904223)>>>0`，`u=s/4294967296`，索引 `floor(u*n_episodes)`；按 CD、PC60、Object 顺序连续使用 RNG。末尾不完整 episode 仍作为完整 cluster，不拆成独立像素/候选。每次所有对照共享抽样索引。

这些区间只条件于现有拟合配置及固定 episode，未校正历史选择、未作多重检验校正，也不模拟重新组成 episode；CD 仅18 clusters，区间本身不稳定。它们支持“CD/Object 的 Confidence 差值不是只有小数点层面的变化”，不支持双视图普遍优于单视图，更不能替代 W1 所需的独立验证。

## 7. 核查范围、证据路径和未验证项

**冻结对象**：`/home/wwww/hzl/results/trace_evolution_review_round02_20261029_0015`。

- `paper/build/main.pdf` SHA256：`1c48c7f1d80d77dae4211dee568dc26d39873ce254a547f1ea1cfdc140f13167`，本轮实际计算，与 packet manifest 一致。完整阅读18页，包括正文、参考文献和附录。
- `paper/build/supplement.pdf` SHA256：`cf12b916c4bb5a7b24fa9514f2951f4d300507629a67322be89ac78781628f4e`，本轮实际计算，与 manifest 一致。完整阅读7页；它通过相同 appendix.tex 编译，未视作额外独立实验。
- 实际渲染查看 main PDF pp.4、9、18，检查目标公式、流程与分割/探测图；其余页通过逐页覆盖的 pdftotext 阅读，并以 TeX 交叉核查。没有以摘要替代全文。

以下定义绝对路径前缀，仅为缩短清单，不表示扫描整棵目录：

- P = `/home/wwww/hzl/results/trace_evolution_review_round02_20261029_0015`
- A = `/home/wwww/hzl/results/trace_self_evolution_20261028_1338`
- B = `/home/wwww/hzl/results/trace_evolution_product_pixel_recovery_20261028_2215`
- C = `/home/wwww/hzl/results/trace_evolution_voc20_cli_recovery_20261028_2312`
- D = `/home/wwww/hzl/results/trace_evolution_final_analysis_20261028_2342`

实际检查路径：

1. `P/packet_manifest.json`；`P/paper/{main,appendix,supplement,preamble}.tex`；两个上述 PDF。
2. `P/paper/generated/{reported_comparison,cd_local,mechanism_status,cd_mechanism,configurations,update_risk,risk_strata,cost_snapshot,seed_replay,native_comparison,comparison_summary}.tex`；`P/paper/generated/{evolution_provenance,provenance,risk_provenance}.json`；`P/paper/figure/final_figure_provenance.json`。旧 native/provenance 内容只作历史来源识别，未混入当前主结果。
3. `P/paper/tools/{export_evolution,export_update_risk}.py`，仅阅读，不执行 exporter 或其导入的 controller。
4. `A/manifest.json`，仅结构化提取运行身份、无本轮搜索/历史开发暴露字段及关键源文件 hash；未读取根目录 contract、audit、old paper 或任何 review。
5. `A/cells/mechanism/{coco_cd,pc60,voc21,coco_object,coco_stuff}/{prediction_protocol.json,result.json,SCORED_SEALED.json,PREDICTIONS_SEALED.json,scores_by_episode.jsonl}`；`C/cells/mechanism/voc20/` 下相同五个文件。独立核查每个策略的 episode pooled counts、mIoU、score 小文件 hash、result→prediction seal 链及 protocol hash，不重读预测 payload。
6. `B/cells/{pc60,coco_object}/` 下相同五个文件，加各自 `replay_gate.json`；`B/runtime/run_cell.py`。
7. `A/cells/repetition/{coco_cd,pc60,voc21,coco_object}/{seed1,seed2}/PREDICTIONS_SEALED.json`；`C/cells/repetition/voc20/{seed1,seed2}/PREDICTIONS_SEALED.json`。仅比较 trace prediction 的封存 hash inventory，不声称重新 hash 了真实 payload。
8. `A/runtime/trace_ovss/method/{two_view,adapter}.py`；`A/runtime/trace_ovss/method/trace/{engine,objective,candidates,interventions}.py`；`A/runtime/trace_ovss/models/dense_clip.py`；`A/runtime/tools/evolution/{run_cell,policies}.py`。这些九个关键文件的实际 hash 与 A manifest 一致。另对 `A/runtime/trace_ovss/data/datasets.py` 和 `resources/class_names/cls_coco_object.txt` 阅读，以及 `A/runtime/tools/trace_algorithm_variants.py` 的有关 quality_only 定义位置查询。
9. `D/{summary.json,FINAL_ANALYSIS_SEALED.json}`；`D/strata_{coco_cd,pc60,voc21,coco_object,coco_stuff}/summary.json`；`D/coco_cd/summary.json`；D final seal 引用的五个 `ANALYSIS_SEALED.json` 和五个 `STRATIFIED_ANALYSIS_SEALED.json` 作小文件 hash 核对。重建 Table 4 五个 split 的 CW/WC 比率；核对 Table 8 summary 的方向/数值。未读取 candidate_transitions.csv，也未从底层候选记录重新计算所有分层统计。
10. 外部一手材料：[MLMP NeurIPS 2025 正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/6c5b82193c5d8e6aa5806239676ddc97-Paper-Conference.pdf) p.9 Table 5、Appendix D；[arXiv v2 全文](https://arxiv.org/html/2505.21844v2) 的相关实现说明；[官方 main.py](https://github.com/dosowiechi/MLMP/blob/main/main.py) 的数据/指标调用；上述官方数据集与指标源文件。核查数值与公开定义，不据其当前 main 推定历史论文执行 commit 完全一致。

**限制**：未运行模型或 GPU；未读取巨大 full/per_image_counts.json；未扫描/重 hash 所有预测、模型或原始图像；未独立核对全部 annotation 的语义映射；未重跑原生基线，CD 原生行依赖冻结包 exporter/provenance 记录；未穷尽查新。数值一致不等于实验从未发生选择偏差，seal 是一致性证据而不是标签访问的完整取证证明。

未读取之前的 review、作者内部 outline/claim ledger、自演化 contract/handoff。历史记忆未作为论文科学结论或分数依据。没有目标分数或预设接收意见。未修改冻结包、实验或其他文件。

## 8. 最终接收判断

当前可以成立的结论是：**在已冻结配置与 episode 下，候选级上下文探测分配在 CD/Object 上优于初始 Confidence/Uniform，区域平均在 PC60/Object 上优于 q 加权 token 归一化。**

当前不能充分成立的更强结论是：这套双视图 min 设计相对更简单探测规则具有稳定的新增价值，或跨论文排名已经确立独立、可迁移的方法优势。接收所缺的是选择过程和验证边界的可审计说明，以及与贡献规模相称的双视图增量证据；不是再增加确定性 seed 重放、挑选更多图例或给代数性质附加可靠性含义。由此给出 **5/10，Borderline，倾向拒稿，置信度4/5**。
